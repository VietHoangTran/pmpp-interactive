import { useEffect, useId, useMemo, useRef, useState, type MouseEvent as RMouseEvent, type PointerEvent as RPointerEvent, type ReactNode } from 'react';
import { Bench } from '../../components/notes';
import { num, useReports, type Reports } from '../../lib/reports';
import { SEGMENT_MODELS, type SegmentModelId } from '../../sim/coalesce';
import {
  BOOK_OPS_F4,
  KERNELS,
  KERNEL_ORDER,
  arithmetic,
  f32text,
  mapKernel,
  movedAtom,
  opsCoarsened,
  opsGather,
  seededAtoms,
  writeStatements,
  type Atom,
  type KernelId,
  type MapResult,
  type Ownership,
  type ThreadInfo,
} from '../../sim/dcs';
import {
  acceptMasks,
  acceptedShare,
  binAtoms,
  binCentre,
  binOf,
  blockWalk,
  capacityFor,
  counts,
  diverges,
  makeWorld,
  maxOccupancy,
  nearFar,
  runGrid,
  superRadius,
  worldH,
  worldW,
  type RadiusMode,
} from '../../sim/cutoff';

// "Who writes where": one explorer with three views, on the models in src/sim/dcs.ts and
// src/sim/cutoff.ts. Section 21.2 uses the mapping view (twice: scatter and gather, then the
// coarsened kernels) and the write-statement view; section 21.5 uses the cutoff view.
// Nothing here computes kernel logic of its own: every number comes from the models.

export type ExplorerView = 'mapping' | 'writes' | 'cutoff';

const thousands = (v: number) => v.toLocaleString('en-US');
const pct = (v: number, digits = 0) => `${(v * 100).toFixed(digits)} %`;
const plural = (n: number, one: string, many = one + 's') => `${thousands(n)} ${n === 1 ? one : many}`;
/** Table cells that may wrap (the shared .threads style keeps cells on one line). */
const WRAP = { whiteSpace: 'normal' as const, verticalAlign: 'top' as const };
/** Below this width a drawing scrolls sideways inside the bench instead of shrinking past legibility. */
const MIN_DRAW = 520;

function Field({ label, children }: { label: ReactNode; children: ReactNode }) {
  return (
    <label className="field">
      {label}
      {children}
    </label>
  );
}

// ================================================================ view 1: who writes where

const KERNEL_SHORT: Record<KernelId, string> = {
  scatter: 'scatter (21.5)',
  gather: 'gather (21.6)',
  printed: 'coarsened, as printed (21.8)',
  adjacent: 'coarsened, adjacent (21.8 fixed)',
  strided: 'coarsened, stride (21.10)',
};

const threadName = (t: ThreadInfo) => `block (${t.bx}, ${t.by}), thread (${t.tx}, ${t.ty})`;

/** The thread's index arithmetic, with its numbers. */
function indexText(r: MapResult, t: ThreadInfo): string {
  const { kernel, blockX: B, blockY: By, W } = r.config;
  const F = r.config.F;
  const pts = t.points.map((p) => `(${p % W}, ${Math.floor(p / W)})`);
  const list = pts.length > 6 ? `${pts.slice(0, 5).join(', ')}, … ${pts.length} points` : pts.join(', ');
  const jText = `j = blockIdx.y·blockDim.y + threadIdx.y = ${t.by}·${By} + ${t.ty} = ${t.j}`;
  const out = t.outside ? ` ${plural(t.outside, 'write')} would fall outside the slice (a bound check stops ${t.outside === 1 ? 'it' : 'them'}).` : '';
  switch (kernel) {
    case 'scatter':
      return t.atom! < 0
        ? `n = (blockIdx.x·blockDim.x + threadIdx.x)·4 = (${t.bx}·${B} + ${t.tx})·4 = ${(t.bx * B + t.tx) * 4}: past the chunk's last atom, so the bound check returns and this thread does nothing.`
        : `n = (blockIdx.x·blockDim.x + threadIdx.x)·4 = (${t.bx}·${B} + ${t.tx})·4 = ${t.atom! * 4}: atom ${t.atom}. It reads that one atom and adds to every one of the ${W * r.config.H} points.`;
    case 'gather':
      return t.points.length
        ? `i = blockIdx.x·blockDim.x + threadIdx.x = ${t.bx}·${B} + ${t.tx} = ${t.i}; ${jText}. It reads every atom and writes point ${list} once.`
        : `i = ${t.bx}·${B} + ${t.tx} = ${t.i}, j = ${t.j}: outside the slice. The bound check returns (the book pads the slice instead, Fig. 21.9).`;
    case 'printed':
      return `i = blockIdx.x·blockDim.x·${F} + threadIdx.x = ${t.bx}·${B}·${F} + ${t.tx} = ${t.i} (threadIdx.x not scaled); ${jText}. It writes i, i+1, …: ${list || 'nothing'}.${out}`;
    case 'adjacent':
      return `i = (blockIdx.x·blockDim.x + threadIdx.x)·${F} = (${t.bx}·${B} + ${t.tx})·${F} = ${t.i}; ${jText}. It writes i to i+${F - 1}: ${list || 'nothing'}.${out}`;
    case 'strided':
      return `i = blockIdx.x·blockDim.x·${F} + threadIdx.x = ${t.bx}·${B}·${F} + ${t.tx} = ${t.i}; ${jText}. It writes i, i+${B}, …: ${list || 'nothing'}.${out}`;
  }
}

function MappingView({ initialKernel }: { initialKernel: KernelId }) {
  const hatch = useId();
  const svgRef = useRef<SVGSVGElement>(null);
  const [kernel, setKernel] = useState<KernelId>(initialKernel);
  const [W, setW] = useState(16);
  const [H, setH] = useState(4);
  const [nAtoms, setNAtoms] = useState(5);
  const [seed, setSeed] = useState(1);
  const [moved, setMoved] = useState<Record<number, Atom>>({});
  const [blockX, setBlockX] = useState(4);
  const [blockY, setBlockY] = useState(2);
  const [F, setF] = useState(4);
  const [atomic, setAtomic] = useState(true);
  const [point, setPoint] = useState(3);
  const [thread, setThread] = useState<number | null>(null);
  const [drag, setDrag] = useState<{ k: number; moved: boolean } | null>(null);

  const atoms = useMemo(() => seededAtoms(nAtoms, W, H, seed).map((a, k) => moved[k] ?? a), [nAtoms, W, H, seed, moved]);
  const res = useMemo(() => mapKernel({ kernel, W, H, atoms, blockX, blockY, F, atomic }), [kernel, W, H, atoms, blockX, blockY, F, atomic]);
  const info = KERNELS[kernel];
  const p = Math.min(point, W * H - 1);
  const writers = res.writers[p];
  const tSel = thread != null && thread < res.threads.length ? res.threads[thread] : null;
  const reset = () => {
    setMoved({});
    setThread(null);
  };

  // Geometry: one cell per point, atoms drawn at their (x, y), labels outside the slice.
  const cs = Math.max(14, Math.min(34, Math.floor(640 / W)));
  const ox = 26;
  const oy = 22;
  const VW = ox + W * cs + 8;
  const VH = oy + H * cs + 8;
  const toSvg = (e: RPointerEvent) => {
    const svg = svgRef.current;
    const m = svg?.getScreenCTM();
    if (!svg || !m) return null;
    const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse());
    return { x: (pt.x - ox) / cs - 0.5, y: (pt.y - oy) / cs - 0.5 };
  };
  const onMove = (e: RPointerEvent) => {
    if (!drag) return;
    const q = toSvg(e);
    if (!q) return;
    const x = Math.max(-0.4, Math.min(W - 0.6, q.x));
    const y = Math.max(-0.4, Math.min(H - 0.6, q.y));
    setMoved((m) => ({ ...m, [drag.k]: movedAtom(atoms[drag.k], x, y) }));
    if (!drag.moved) setDrag({ ...drag, moved: true });
  };
  const pickPoint = (q: number) => {
    setPoint(q);
    setThread(res.writers[q][0] ?? null);
  };
  const selectedAtom = kernel === 'scatter' && tSel ? tSel.atom : undefined;
  const shade = (q: number) => {
    const w = res.writers[q];
    if (kernel === 'scatter' || w.length !== 1) return 0;
    return 0.05 + 0.075 * (res.threads[w[0]].id % 4);
  };
  const maxW = Math.max(...res.writers.map((w) => w.length));
  const minW = Math.min(...res.writers.map((w) => w.length));
  const tolerance = 1e-5 * atoms.reduce((s, a) => s + Math.abs(a.q / Math.hypot(p % W - a.x, Math.floor(p / W) - a.y, a.z)), 0);
  const diff = res.potentials[p] - res.direct[p];

  return (
    <div data-view="mapping" data-kernel={kernel} data-verdict={res.verdict} data-conflicts={res.conflicts}
      data-unwritten={res.unwritten} data-const={res.constAddrsPerWarp} data-writers-max={maxW} data-writers-min={minW}
      data-correct={res.correct ? 'yes' : 'no'}>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label="Kernel" style={{ flexWrap: 'wrap', rowGap: 4 }}>
          {KERNEL_ORDER.map((k) => (
            <button key={k} aria-pressed={k === kernel} onClick={() => { setKernel(k); setThread(null); }} title={`${KERNELS[k].label} (${KERNELS[k].fig})`}>
              {KERNEL_SHORT[k]}
            </button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <Field label={`Slice width ${W}`}>
          <input type="range" min={8} max={40} value={W} aria-label="slice width" onChange={(e) => { setW(+e.target.value); reset(); }} />
        </Field>
        <Field label={`Height ${H}`}>
          <input type="range" min={2} max={8} value={H} aria-label="slice height" onChange={(e) => { setH(+e.target.value); reset(); }} />
        </Field>
        <Field label={`Atoms ${nAtoms}`}>
          <input type="range" min={1} max={12} value={nAtoms} aria-label="atoms" onChange={(e) => { setNAtoms(+e.target.value); reset(); }} />
        </Field>
        <button onClick={() => { setSeed(seed + 1); reset(); }}>New random atoms</button>
        <Field label="blockDim.x">
          <select value={blockX} onChange={(e) => { setBlockX(+e.target.value); setThread(null); }}>
            {[4, 8, 16, 32].map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </Field>
        {kernel !== 'scatter' ? (
          <Field label="blockDim.y">
            <select value={blockY} onChange={(e) => { setBlockY(+e.target.value); setThread(null); }}>
              {[1, 2, 4].map((b) => <option key={b} value={b}>{b}</option>)}
            </select>
          </Field>
        ) : null}
        {info.coarsened ? (
          <Field label="Coarsening factor">
            <select value={F} onChange={(e) => { setF(+e.target.value); setThread(null); }}>
              {[2, 4, 8].map((b) => <option key={b} value={b}>{b}</option>)}
            </select>
          </Field>
        ) : null}
        {kernel === 'scatter' ? (
          <label className="check">
            <input type="checkbox" checked={!atomic} onChange={(e) => setAtomic(!e.target.checked)} /> plain += instead of atomicAdd
          </label>
        ) : null}
      </div>

      <p className="ui" style={{ margin: '0 0 0.6rem' }} data-badges="">
        {res.verdict === 'one-writer' ? <span className="chip ok">one writer per point: no atomic needed</span>
          : res.verdict === 'atomic' ? <span className="chip" style={{ border: '1px solid var(--ink)' }}>atomicAdd needed: every point has {maxW} writers</span>
          : <span className="chip fail">data race: {plural(res.conflicts, 'point')} with several writers and a plain +=</span>}{' '}
        {res.unwritten ? <span className="chip fail">{plural(res.unwritten, 'point')} never written</span> : null}{' '}
        {res.correct ? <span className="chip ok">result equals the direct sum</span> : <span className="chip fail">result wrong at {plural(res.wrong, 'point')}</span>}
      </p>

      <div style={{ overflowX: 'auto' }}>
        <svg ref={svgRef} viewBox={`0 0 ${VW} ${VH}`} role="img" aria-label="Grid points coloured by the thread that writes them, with the atoms above"
          style={{ width: '100%', maxWidth: VW * 1.2, minWidth: Math.min(MIN_DRAW, VW), display: 'block', touchAction: 'none' }}
          onPointerMove={onMove} onPointerUp={() => setDrag(null)} onPointerCancel={() => setDrag(null)}>
          <defs>
            <pattern id={hatch} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1="0" y1="0" x2="0" y2="6" stroke="var(--fail)" strokeWidth="1.4" />
            </pattern>
          </defs>
          {Array.from({ length: W }, (_, i) => (i % (W > 24 ? 4 : 2) === 0 ? (
            <text key={i} x={ox + i * cs + cs / 2} y={oy - 7} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{i}</text>
          ) : null))}
          {Array.from({ length: H }, (_, j) => (
            <text key={j} x={ox - 6} y={oy + j * cs + cs / 2 + 3} fontSize="9" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>{j}</text>
          ))}
          {res.writers.map((w, q) => {
            const i = q % W;
            const j = Math.floor(q / W);
            const n = w.length;
            const race = n > 1 && res.verdict === 'race';
            const mine = tSel && kernel !== 'scatter' && tSel.points.includes(q);
            return (
              <g key={q} data-point={q} data-writers={n} onClick={() => pickPoint(q)} style={{ cursor: 'pointer' }}>
                <rect x={ox + i * cs + 1} y={oy + j * cs + 1} width={cs - 2} height={cs - 2} rx="2"
                  fill={n === 0 ? `url(#${hatch})` : race ? 'var(--fail-wash)' : 'var(--ink)'}
                  fillOpacity={n === 0 || race ? 1 : shade(q)}
                  stroke={race ? 'var(--fail)' : 'var(--rule)'} strokeWidth={race ? 1.4 : 1} />
                {mine ? <rect x={ox + i * cs + 3} y={oy + j * cs + 3} width={cs - 6} height={cs - 6} rx="2" fill="none" stroke="var(--ink)" strokeDasharray="3 2" /> : null}
                {cs >= 18 && n !== 1 ? (
                  <text x={ox + i * cs + cs / 2} y={oy + j * cs + cs / 2 + 3.5} fontSize="9.5" textAnchor="middle"
                    style={{ fill: race || n === 0 ? 'var(--fail)' : 'var(--ink-2)', fontWeight: race ? 650 : 400 }}>×{n}</text>
                ) : null}
                {cs >= 22 && n === 1 && kernel !== 'scatter' ? (
                  <text x={ox + i * cs + cs / 2} y={oy + j * cs + cs / 2 + 3.5} fontSize="8.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
                    t{res.threads[w[0]].tx}
                  </text>
                ) : null}
              </g>
            );
          })}
          <rect x={ox + (p % W) * cs} y={oy + Math.floor(p / W) * cs} width={cs} height={cs} fill="none" stroke="var(--ink)" strokeWidth="2.4" pointerEvents="none" />
          {atoms.map((a, k) => (
            <g key={k} data-atom={k}
              onPointerDown={(e) => { e.stopPropagation(); (e.currentTarget.ownerSVGElement ?? e.currentTarget).setPointerCapture?.(e.pointerId); setDrag({ k, moved: false }); }}
              onPointerUp={() => {
                if (drag && !drag.moved && kernel === 'scatter') setThread(res.threads.find((t) => t.atom === k)?.id ?? null);
                setDrag(null);
              }}
              style={{ cursor: 'grab' }}>
              <circle cx={ox + (a.x + 0.5) * cs} cy={oy + (a.y + 0.5) * cs} r={Math.max(5, cs * 0.3)}
                fill="var(--const)" fillOpacity={selectedAtom === k ? 1 : 0.78}
                stroke={selectedAtom === k ? 'var(--ink)' : 'var(--page)'} strokeWidth={selectedAtom === k ? 2.2 : 1.2} />
              <text x={ox + (a.x + 0.5) * cs} y={oy + (a.y + 0.5) * cs + 3.2} fontSize="9" textAnchor="middle" style={{ fill: 'var(--page)', pointerEvents: 'none' }}>
                {k}
              </text>
            </g>
          ))}
        </svg>
      </div>

      <div className="cols" style={{ marginTop: '0.8rem' }}>
        <div className="ui" style={{ fontSize: '0.86rem' }}>
          <table className="threads" aria-label="What this mapping does">
            <tbody>
              <tr><td style={WRAP}>launch</td><td style={WRAP}>{res.launch.gridX} × {res.launch.gridY} blocks of {res.launch.blockX} × {res.launch.blockY}: {plural(res.launch.threads, 'thread')}{res.idle ? `, ${res.idle} with nothing to do` : ''}</td></tr>
              <tr><td style={WRAP}>writers per point</td><td style={WRAP} data-readout="writers">{minW === maxW ? minW : `${minW} to ${maxW}`}</td></tr>
              <tr><td style={WRAP}>constant addresses per warp instruction</td><td style={WRAP} data-readout="const">{res.constAddrsPerWarp} {kernel === 'scatter' ? '(one per lane: each thread reads its own atom)' : '(every lane reads the same atom)'}</td></tr>
              <tr><td style={WRAP}>pairs per busy thread</td><td style={WRAP}>{thousands(res.pairsPerThread)}</td></tr>
            </tbody>
          </table>
        </div>
        <div className="ui" style={{ fontSize: '0.86rem' }} data-selected-point={p}>
          <p style={{ margin: '0 0 0.4rem' }}>
            <b>Point ({p % W}, {Math.floor(p / W)})</b>: {writers.length === 0 ? 'no thread writes it' : `written by ${plural(writers.length, 'thread')}`}.{' '}
            Kernel's result <span className="mono" data-readout="result">{f32text(res.potentials[p])}</span>; float64 direct sum{' '}
            <span className="mono" data-readout="direct">{res.direct[p].toPrecision(9)}</span>
            {Math.abs(diff) <= tolerance ? <> (difference {diff.toExponential(1)}: float32 rounding)</> : <b className="t-fail"> (wrong)</b>}.
          </p>
          {writers.length ? (
            <div className="row" style={{ gap: '0.3rem', marginBottom: '0.4rem' }}>
              {writers.slice(0, 12).map((id) => (
                <button key={id} className="quiet" aria-pressed={thread === id} onClick={() => setThread(id)} style={{ fontSize: '0.78rem', padding: '0.15rem 0.45rem' }}>
                  {kernel === 'scatter' ? `atom ${res.threads[id].atom}` : `(${res.threads[id].bx},${res.threads[id].by}|${res.threads[id].tx},${res.threads[id].ty})`}
                </button>
              ))}
              {writers.length > 12 ? <span className="muted">…</span> : null}
            </div>
          ) : null}
          <p className="narrate" style={{ margin: 0, minHeight: '2.6em' }} data-thread-text="">
            {tSel ? <><b>{threadName(tSel)}</b> (warp {tSel.warp}, lane {tSel.lane}): {indexText(res, tSel)}</> : kernel === 'scatter'
              ? 'Click an atom to see its thread, or a point to list the threads that write it.'
              : 'Click a point to list the threads that write it, then pick one to see its index arithmetic.'}
          </p>
        </div>
      </div>
      {info.coarsened ? <OpsPanel F0={F} /> : null}
    </div>
  );
}

/** Operations per atom: F threads of Fig. 21.6 against one coarsened thread, counted from the listings. */
function OpsPanel({ F0 }: { F0: number }) {
  const [F, setF] = useState(F0);
  useEffect(() => setF(F0), [F0]);
  const g = opsGather(F);
  const c = opsCoarsened(F);
  const at4 = F === 4;
  const rows: { name: string; g: ReactNode; c: ReactNode; bg: ReactNode; bc: ReactNode; key: string }[] = [
    { key: 'loads', name: 'constant-memory loads', g: g.loads, c: c.loads, bg: BOOK_OPS_F4.gather.loads, bc: BOOK_OPS_F4.coarsened.loads },
    { key: 'sub', name: 'subtractions', g: g.sub, c: c.sub, bg: BOOK_OPS_F4.gather.sub, bc: BOOK_OPS_F4.coarsened.sub },
    { key: 'add', name: 'additions', g: g.add, c: c.add, bg: BOOK_OPS_F4.gather.add, bc: BOOK_OPS_F4.coarsened.add },
    { key: 'mul', name: 'multiplications', g: g.mul, c: <>{c.mul} <span className="muted">({c.mulUnhoisted} as written)</span></>, bg: BOOK_OPS_F4.gather.mul, bc: BOOK_OPS_F4.coarsened.mul },
    { key: 'div', name: 'divisions', g: g.div, c: c.div, bg: BOOK_OPS_F4.gather.div, bc: BOOK_OPS_F4.coarsened.div },
    { key: 'sqrt', name: 'square roots', g: g.sqrt, c: c.sqrt, bg: 'not counted', bc: 'not counted' },
    { key: 'branch', name: 'loop branches', g: g.branch, c: c.branch, bg: '–', bc: '–' },
    { key: 'total', name: 'arithmetic, without roots', g: arithmetic(g), c: <>{arithmetic(c, true)} <span className="muted">({arithmetic(c)} as written)</span></>, bg: BOOK_OPS_F4.gather.total, bc: BOOK_OPS_F4.coarsened.total },
  ];
  return (
    <div className="panel" style={{ marginTop: '1rem' }} data-ops={F}>
      <div className="row" style={{ marginBottom: '0.5rem' }}>
        <b className="ui">Work per atom for {F} {F === 1 ? 'point' : 'points'}</b>
        <Field label={`F = ${F}`}>
          <input type="range" min={1} max={8} value={F} aria-label="points per coarsened thread" onChange={(e) => setF(+e.target.value)} />
        </Field>
      </div>
      <table className="data" style={{ margin: 0 }}>
        <thead>
          <tr>
            <th>per atom</th>
            <th className="num">{F} threads of Fig. 21.6</th>
            <th className="num">one coarsened thread</th>
            <th className="num" style={{ opacity: at4 ? 1 : 0.5 }}>book, Fig. 21.6 ×4</th>
            <th className="num" style={{ opacity: at4 ? 1 : 0.5 }}>book, Fig. 21.8</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} data-op={r.key}>
              <td>{r.name}</td>
              <td className="num" data-g="">{r.g}</td>
              <td className="num" data-c="">{r.c}</td>
              <td className="num" style={{ opacity: at4 ? 1 : 0.5 }}>{r.bg}</td>
              <td className="num" style={{ opacity: at4 ? 1 : 0.5 }}>{r.bc}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="ui muted" style={{ margin: '0.5rem 0 0', fontSize: '0.8rem' }}>
        Counted from the listings: "hoisted" treats the m·gridspacing products as computed once, outside the atom loop.
        The book's columns are its printed figures for four points (p. 521); they are greyed when F is not 4.
      </p>
    </div>
  );
}

// ================================================================ view 2: write statements

/** The SIMD width a lab recorded on this Mac: lab 21.1's group_width if it has run, else any lab's. */
export function appleWidth(reports: Reports): { width: number; source: string } | null {
  const run = reports.metal;
  if (!run) return null;
  const rows = run.results.flatMap((r) => r.bench.map((b) => ({ lab: r.lab, b })));
  const mine = rows.find((x) => x.lab === 'ch21/lab1_dcs' && num(x.b.group_width));
  if (mine) return { width: num(mine.b.group_width)!, source: 'lab 21.1 on this Mac' };
  for (const x of rows) {
    const w = num(x.b.simd_width) ?? (x.b.name === 'simd_width' ? num(x.b.width) : undefined) ?? num(x.b.threadExecutionWidth);
    if (w) return { width: w, source: `lab ${x.lab.replace(/^ch0?(\d+)\/lab(\d+).*/, '$1.$2')} on this Mac (lab 21.1 has not run yet)` };
  }
  return null;
}

type Group = '32' | '64' | 'apple';
const OWNERSHIP: { id: Ownership; label: string }[] = [
  { id: 'adjacent', label: 'adjacent (Fig. 21.8, corrected)' },
  { id: 'strided', label: 'blockDim.x stride (Fig. 21.10)' },
  { id: 'float4', label: 'one float4 store (Exercise 4)' },
];

function WritesView() {
  const { reports } = useReports();
  const apple = appleWidth(reports);
  const [own, setOwn] = useState<Ownership>('adjacent');
  const [group, setGroup] = useState<Group>('32');
  const [appleChoice, setAppleChoice] = useState(32);
  const [blockX, setBlockX] = useState(128);
  const [blockY, setBlockY] = useState(1);
  const [seg, setSeg] = useState<SegmentModelId>('nvidia');
  const [appleSeg, setAppleSeg] = useState(32);
  const [F, setF] = useState(4);
  const [m, setM] = useState(0);
  const [playing, setPlaying] = useState(false);

  const warpWidth = group === 'apple' ? (apple?.width ?? appleChoice) : Number(group);
  const segmentBytes = SEGMENT_MODELS[seg].segmentBytes ?? appleSeg;
  const unit = seg === 'nvidia' ? 'sector' : seg === 'metal' ? 'segment' : 'line';
  const Fe = own === 'float4' && F % 4 !== 0 ? 4 : F;
  const res = useMemo(
    () => writeStatements({ ownership: own, F: Fe, warpWidth, blockX, blockY, rowPitch: 1024, segmentBytes }),
    [own, Fe, warpWidth, blockX, blockY, segmentBytes],
  );
  const n = res.statements.length;
  const at = Math.min(m, n - 1);
  const cur = res.statements[at];
  const soFar = res.statements.slice(0, at + 1).reduce((s, x) => s + x.access.transactions, 0);
  useEffect(() => setM(0), [own, Fe, warpWidth, blockX, blockY, segmentBytes]);
  useEffect(() => {
    if (!playing) return;
    if (at >= n - 1) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setM((x) => x + 1), 900);
    return () => window.clearTimeout(id);
  }, [playing, at, n]);

  // One strip per slice row the warp touches; each covers every segment any statement touches there.
  const rowsTouched = [...new Set(res.statements.flatMap((s) => s.lanes.map((l) => l.ty)))].sort((a, b) => a - b);
  const bpl = cur.bytesPerLane;
  const strips = rowsTouched.map((ty) => {
    const all = res.statements.flatMap((s) => s.lanes.filter((l) => l.ty === ty).map((l) => l.byte));
    const lo = Math.floor(Math.min(...all) / segmentBytes) * segmentBytes;
    const hi = Math.ceil((Math.max(...all) + bpl) / segmentBytes) * segmentBytes;
    return { ty, lo, hi };
  });
  const span = Math.max(...strips.map((s) => s.hi - s.lo));
  const px = Math.min(1.6, 660 / span);
  const X0 = 6;
  const pitch = 54;
  const VW = Math.max(360, Math.ceil(X0 * 2 + span * px));
  const VH = 6 + strips.length * pitch;
  const curSegs = new Set(cur.access.segments);
  const lanes = cur.lanes;
  const step = lanes.length > 1 ? lanes[1].element - lanes[0].element : 0;

  return (
    <div data-view="writes" data-statement={at} data-statements={n} data-transactions={cur.access.transactions}
      data-sofar={soFar} data-total={res.transactions} data-runs={cur.runs} data-width={warpWidth} data-segment={segmentBytes}>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label="Ownership" style={{ flexWrap: 'wrap', rowGap: 4 }}>
          {OWNERSHIP.map((o) => (
            <button key={o.id} aria-pressed={o.id === own} onClick={() => setOwn(o.id)}>{o.label}</button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <Field label="Group width">
          <select value={group} onChange={(e) => setGroup(e.target.value as Group)}>
            <option value="32">32 lanes (NVIDIA; AMD Radeon)</option>
            <option value="64">64 lanes (AMD Instinct)</option>
            <option value="apple">Apple: {apple ? `${apple.width} lanes, from ${apple.source}` : 'not recorded here yet'}</option>
          </select>
        </Field>
        {group === 'apple' && !apple ? (
          <Field label="SIMD width (your choice)">
            <select value={appleChoice} onChange={(e) => setAppleChoice(+e.target.value)}>
              {[16, 32, 64].map((w) => <option key={w} value={w}>{w}</option>)}
            </select>
          </Field>
        ) : null}
        <Field label="blockDim.x">
          <select value={blockX} onChange={(e) => setBlockX(+e.target.value)}>
            {[16, 32, 64, 128].map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </Field>
        <Field label="blockDim.y">
          <select value={blockY} onChange={(e) => setBlockY(+e.target.value)}>
            {[1, 2, 4].map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </Field>
        <Field label="Coarsening factor">
          <select value={Fe} onChange={(e) => setF(+e.target.value)}>
            {[2, 4, 8].map((b) => <option key={b} value={b} disabled={own === 'float4' && b % 4 !== 0}>{b}</option>)}
          </select>
        </Field>
        <Field label="Memory unit">
          <select value={seg} onChange={(e) => setSeg(e.target.value as SegmentModelId)}>
            <option value="nvidia">NVIDIA: 32-byte sectors</option>
            <option value="amd64">AMD: 64-byte lines</option>
            <option value="amd128">AMD: 128-byte lines</option>
            <option value="metal">Apple: not documented</option>
          </select>
        </Field>
        {seg === 'metal' ? (
          <Field label="Segment (your choice)">
            <select value={appleSeg} onChange={(e) => setAppleSeg(+e.target.value)}>
              {[32, 64, 128].map((b) => <option key={b} value={b}>{b} bytes</option>)}
            </select>
          </Field>
        ) : null}
      </div>
      {group === 'apple' && !apple ? (
        <p className="ui muted" style={{ margin: '0 0 0.6rem', fontSize: '0.82rem' }}>
          Apple documents no fixed SIMD width: a kernel reads it from its pipeline (threadExecutionWidth). Run lab 21.1 on
          this Mac and the value it records replaces your choice.
        </p>
      ) : null}

      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <button onClick={() => { setPlaying(false); setM(0); }} disabled={at === 0}>Restart</button>
        <button onClick={() => { setPlaying(false); setM(at - 1); }} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => { setPlaying(false); setM(at + 1); }} disabled={at >= n - 1}>Next statement</button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= n - 1}>{playing ? 'Pause' : 'Play'}</button>
        <span className="muted" aria-live="polite">{own === 'float4' ? 'store' : 'statement'} {at + 1} of {n}</span>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${VW} ${VH}`} role="img" aria-label="Bytes of the slice one warp writes in one statement"
          style={{ width: '100%', maxWidth: VW * 1.2, minWidth: Math.min(MIN_DRAW, VW), display: 'block' }}>
          {strips.map((s, r) => {
            const y = 4 + r * pitch;
            const segs: number[] = [];
            for (let b = s.lo; b < s.hi; b += segmentBytes) segs.push(b / segmentBytes);
            return (
              <g key={s.ty} data-strip={s.ty}>
                <text x={X0} y={y + 10} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
                  row {s.ty} of the block <tspan style={{ fill: 'var(--ink-3)' }}>bytes {thousands(s.lo)} to {thousands(s.hi - 1)}</tspan>
                </text>
                {segs.map((g) => (
                  <rect key={g} x={X0 + (g * segmentBytes - s.lo) * px} y={y + 15} width={Math.max(1, segmentBytes * px - 1)} height={22}
                    fill={curSegs.has(g) ? 'var(--global-wash)' : 'none'} stroke={curSegs.has(g) ? 'var(--global)' : 'var(--rule)'} strokeWidth="0.8"
                    data-seg={curSegs.has(g) ? 'on' : 'off'} />
                ))}
                {res.statements.slice(0, at).flatMap((st) => st.lanes.filter((l) => l.ty === s.ty).map((l) => (
                  <rect key={`${st.m}-${l.lane}`} x={X0 + (l.byte - s.lo) * px} y={y + 19} width={Math.max(0.8, st.bytesPerLane * px - 0.3)} height={14}
                    fill="var(--global)" fillOpacity="0.25" />
                )))}
                {lanes.filter((l) => l.ty === s.ty).map((l) => {
                  const w = Math.max(1, bpl * px - 0.3);
                  return (
                    <g key={l.lane}>
                      <rect x={X0 + (l.byte - s.lo) * px} y={y + 19} width={w} height={14} fill="var(--global)">
                        <title>lane {l.lane}: element {l.element}, bytes {thousands(l.byte)} to {thousands(l.byte + bpl - 1)}</title>
                      </rect>
                      {w >= 12 ? <text x={X0 + (l.byte - s.lo) * px + w / 2} y={y + 29} fontSize="7.5" textAnchor="middle" style={{ fill: 'var(--page)' }}>{l.lane}</text> : null}
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
      </div>

      <p className="ui" style={{ margin: '0.6rem 0 0.3rem' }} data-readout="statement">
        {own === 'float4' ? `Store ${at + 1}` : `Statement ${at + 1}`}: lane l writes element {lanes[0].element}
        {step ? <> + {step}·l</> : null}
        {rowsTouched.length > 1 ? <> (lanes past blockDim.x continue on the next row)</> : null}, {bpl} bytes each.{' '}
        <b>{plural(cur.access.transactions, unit)}</b>, {thousands(cur.access.bytesNeeded)} of {thousands(cur.access.bytesFetched)} bytes
        used ({pct(cur.access.efficiency)}), {plural(cur.runs, 'run')} of contiguous bytes.
      </p>
      <p className="ui muted" style={{ margin: 0 }} data-readout="total">
        So far: {plural(soFar, unit)} over {at + 1} of {n} {own === 'float4' ? 'stores' : 'statements'}. All of them:{' '}
        {plural(res.transactions, unit)}, {thousands(res.bytesNeeded)} bytes written, {thousands(res.bytesFetched)} moved.
        {own === 'float4' && !res.aligned ? <b className="t-fail"> Misaligned: a float4 needs every row to start on a 16-byte boundary.</b> : null}
      </p>
    </div>
  );
}

// ================================================================ view 3: cutoff and bins

const CUT_3D = [8, 12, 16];
const BIN_3D = [2, 4, 8];

function CutoffView() {
  const hatch = useId();
  const svgRef = useRef<SVGSVGElement>(null);
  const [side, setSide] = useState(4);
  const [bins, setBins] = useState(8);
  const [density, setDensity] = useState(2);
  const [seed, setSeed] = useState(1);
  const [c, setC] = useState(1.3);
  const [mode, setMode] = useState<RadiusMode>('half');
  const [cap, setCap] = useState<number | null>(null);
  const [block, setBlock] = useState<[number, number]>([3, 3]);
  const [thread, setThread] = useState(5);
  /** The staged atom shown on the lanes; null until the reader picks one: then the first atom that splits a warp. */
  const [pick, setPick] = useState<{ entry: number; slot: number } | null>(null);
  const [abs, setAbs] = useState(false);
  const [prev, setPrev] = useState<{ points: number; atoms: number; direct: number; tests: number; perPoint: number } | null>(null);
  const [c3, setC3] = useState(12);
  const [b3, setB3] = useState(4);

  const nAtoms = Math.round(density * bins * bins);
  const world = useMemo(() => makeWorld(bins, bins, side, nAtoms, seed), [bins, side, nAtoms, seed]);
  const maxOcc = maxOccupancy(world);
  const capAuto = useMemo(() => capacityFor(world, 0.03), [world]);
  const capacity = Math.max(1, Math.min(maxOcc || 1, cap ?? capAuto));
  const binning = useMemo(() => binAtoms(world, capacity), [world, capacity]);
  const cutoff = c * side;
  const R = superRadius(cutoff, side, 2, mode) / side;
  const bx = Math.min(block[0], bins - 1);
  const by = Math.min(block[1], bins - 1);
  const walk = useMemo(() => blockWalk(world, binning, cutoff, R, bx, by), [world, binning, cutoff, R, bx, by]);
  const grid = useMemo(() => runGrid(world, binning, cutoff, R, false), [world, binning, cutoff, R]);
  const cnt = counts(R, 2);
  const t = Math.min(thread, walk.points.length - 1);
  const [pi, pj] = walk.points[t];
  const nf = nearFar(world, pi, pj, cutoff);
  const inBox = walk.list.filter((e) => e.inBox);
  const auto = useMemo(() => {
    for (const x of walk.list)
      for (let k = 0; k < x.slots.length; k++) if (x.slots[k] >= 0 && acceptMasks(x, k, 32).some(diverges)) return { entry: x.k, slot: k };
    const first = walk.list.find((x) => x.inBox);
    return { entry: first ? first.k : 0, slot: 0 };
  }, [walk]);
  const chosen = pick ?? auto;
  const e = walk.list[Math.min(chosen.entry, walk.list.length - 1)];
  const sl = Math.min(chosen.slot, Math.max(0, e.slots.length - 1));
  const masks = e.slots.length ? acceptMasks(e, sl, 32) : [];
  const threadStats = walk.list.reduce((s, x) => {
    for (const row of x.outcome) {
      const o = row[t];
      s.scanned++;
      if (o === 'accept') s.accepted++;
      if (o === 'dummy') s.dummy++;
    }
    return s;
  }, { scanned: 0, accepted: 0, dummy: 0 });
  const overflowSet = new Set(binning.overflow);
  const acceptedByThread = new Set<number>();
  for (const x of walk.list) x.slots.forEach((s, k) => { if (s >= 0 && x.outcome[k][t] === 'accept') acceptedByThread.add(s); });
  const listBins = new Set(inBox.map((x) => `${x.bx},${x.by}`));
  const points = worldW(world) * worldH(world);
  const stats = { points, atoms: world.atoms.length, direct: grid.directPairs, tests: grid.tests, perPoint: grid.tests / points };

  // Drawing: the box, bins, the block's points, atoms, the super circle and the thread's cutoff circle.
  const BOX = 380;
  const bp = BOX / bins; // px per bin
  const gp = bp / side; // px per grid unit
  const O = 8;
  const toPx = (x: number) => O + (x + 0.5) * gp;
  const [cx, cy] = binCentre(world, bx, by);
  const onClick = (ev: RMouseEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    const m = svg?.getScreenCTM();
    if (!svg || !m) return;
    const q = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(m.inverse());
    const gx = (q.x - O) / gp - 0.5;
    const gy = (q.y - O) / gp - 0.5;
    const [nbx, nby] = binOf(world, gx, gy);
    if (nbx === bx && nby === by) {
      const tx = Math.max(0, Math.min(side - 1, Math.round(gx) - bx * side));
      const ty = Math.max(0, Math.min(side - 1, Math.round(gy) - by * side));
      setThread(ty * side + tx);
    } else {
      setBlock([nbx, nby]);
      setPick(null);
    }
  };
  const share3 = acceptedShare(c3, b3, 3, 'half');
  const n3h = counts(superRadius(c3, b3, 3, 'half') / b3, 3);
  const n3f = counts(superRadius(c3, b3, 3, 'full') / b3, 3);

  return (
    <div data-view="cutoff" data-list={cnt.total} data-full={cnt.full} data-partial={cnt.partial} data-radius={R.toFixed(3)}
      data-capacity={capacity} data-overflow={binning.overflow.length} data-dummy={binning.dummy} data-tests={grid.tests}
      data-direct={grid.directPairs} data-thread-scanned={threadStats.scanned} data-thread-accepted={threadStats.accepted}
      data-bins={bins} data-atoms={world.atoms.length}>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <Field label={`Cutoff ${c.toFixed(2)} bin widths`}>
          <input type="range" min={0.5} max={3} step={0.05} value={c} aria-label="cutoff in bin widths" onChange={(ev) => setC(+ev.target.value)} />
        </Field>
        <button onClick={() => { setC(Number((2 - Math.SQRT2 / 2).toFixed(4))); setMode('half'); }}>R = 2 bin widths</button>
        <Field label="Super-circle radius">
          <select value={mode} onChange={(ev) => setMode(ev.target.value as RadiusMode)}>
            <option value="half">cutoff + half the diagonal (text, Fig. 21.14)</option>
            <option value="full">cutoff + the whole diagonal (Fig. 21.13's label)</option>
          </select>
        </Field>
        <Field label="Bin side">
          <select value={side} onChange={(ev) => { setSide(+ev.target.value); setThread(0); setPrev(null); }}>
            <option value={4}>4 grid spacings (16 threads a block)</option>
            <option value={8}>8 grid spacings (64 threads a block)</option>
          </select>
        </Field>
        <Field label={`Atoms per bin ${density} (${thousands(nAtoms)} atoms)`}>
          <input type="range" min={0.5} max={4} step={0.25} value={density} aria-label="atoms per bin" onChange={(ev) => { setDensity(+ev.target.value); setCap(null); setPrev(null); }} />
        </Field>
        <button onClick={() => { setSeed(seed + 1); setCap(null); }}>New random atoms</button>
        {bins < 16 ? (
          <button onClick={() => { setPrev(stats); setBins(bins * 2); setCap(capacity); }}>Double the box</button>
        ) : (
          <button onClick={() => { setPrev(null); setBins(bins / 2); setCap(capacity); setBlock([Math.min(bx, bins / 2 - 1), Math.min(by, bins / 2 - 1)]); }}>Back to {bins / 2} × {bins / 2} bins</button>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 5fr) minmax(0, 4fr)' }}>
        <div style={{ overflowX: 'auto' }}>
          <svg ref={svgRef} viewBox={`0 0 ${BOX + 2 * O} ${BOX + 2 * O}`} role="img" aria-label="The box of bins, the chosen block's neighbourhood and the atoms"
            style={{ width: '100%', maxWidth: 470, minWidth: 300, display: 'block', cursor: 'pointer' }} onClick={onClick}>
            <defs>
              <pattern id={hatch} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1" />
              </pattern>
            </defs>
            {Array.from({ length: bins * bins }, (_, k) => {
              const x = k % bins;
              const y = Math.floor(k / bins);
              const ent = walk.list.find((q) => q.bx === x && q.by === y);
              const isBlock = x === bx && y === by;
              const isEntry = e.inBox && e.bx === x && e.by === y;
              return (
                <rect key={k} x={O + x * bp} y={O + y * bp} width={bp} height={bp}
                  fill={ent ? 'var(--shared)' : 'var(--cell)'} fillOpacity={ent ? (ent.rel.cover === 'full' ? 0.32 : 0.13) : 1}
                  stroke={isEntry ? 'var(--ink)' : 'var(--rule)'} strokeWidth={isEntry ? 2 : isBlock ? 1.6 : 0.6}
                  data-bin={`${x},${y}`} data-cover={ent ? ent.rel.cover : 'none'} />
              );
            })}
            <rect x={O + bx * bp} y={O + by * bp} width={bp} height={bp} fill="none" stroke="var(--ink)" strokeWidth="2.4" pointerEvents="none" />
            <circle cx={toPx(cx)} cy={toPx(cy)} r={R * bp} fill="none" stroke="var(--ink)" strokeWidth="1.4" strokeDasharray="6 4" pointerEvents="none" data-super="" />
            <circle cx={toPx(pi)} cy={toPx(pj)} r={cutoff * gp} fill="none" stroke="var(--ink-2)" strokeWidth="1" pointerEvents="none" />
            {walk.points.map(([i, j], k) => (
              <circle key={k} cx={toPx(i)} cy={toPx(j)} r={k === t ? 3.4 : 1.4} fill="var(--ink)" pointerEvents="none" />
            ))}
            {world.atoms.map((a, k) => {
              const home = binOf(world, a.x, a.y).join(',');
              const scanned = listBins.has(home) && !overflowSet.has(k);
              const ok = acceptedByThread.has(k);
              return (
                <circle key={k} cx={toPx(a.x)} cy={toPx(a.y)} r={ok ? 3.2 : scanned ? 2.3 : 1.8}
                  fill={scanned ? 'var(--global)' : 'var(--ink-3)'} fillOpacity={ok || scanned ? 1 : 0.55}
                  stroke={overflowSet.has(k) ? 'var(--fail)' : ok ? 'var(--ink)' : 'none'} strokeWidth={overflowSet.has(k) ? 1.4 : 1}
                  pointerEvents="none" />
              );
            })}
          </svg>
          <p className="ui muted" style={{ fontSize: '0.78rem', margin: '0.4rem 0 0' }}>
            Dashed: the super circle around the chosen block (outlined). Thin: the cutoff around the chosen thread's point.
            Darker bins are fully inside the super circle, lighter ones partly. Large dots: atoms the thread accepts; red
            rings: atoms on the overflow list. Click a bin to choose its block, a point inside the chosen block to choose
            its thread.
          </p>
        </div>

        <div className="ui" style={{ fontSize: '0.86rem' }}>
          <p style={{ margin: '0 0 0.5rem' }} data-readout="list">
            R = {c.toFixed(2)} + {mode === 'half' ? '0.71' : '1.41'} = <b>{R.toFixed(2)} bin widths</b>. The list has{' '}
            <b>{cnt.total} bins</b>: {cnt.full} fully inside, {cnt.partial} partly.
            {inBox.length < cnt.total ? <> For this block {cnt.total - inBox.length} of them lie outside the box and are skipped.</> : null}
          </p>
          <div className="row" style={{ gap: '0.25rem', marginBottom: '0.3rem' }} role="group" aria-label="The list">
            {walk.list.map((x) => (
              <button key={x.k} className="quiet" aria-pressed={x.k === e.k} disabled={!x.inBox}
                onClick={() => setPick({ entry: x.k, slot: 0 })} style={{ fontSize: '0.72rem', padding: '0.1rem 0.35rem' }}>
                {abs ? `(${x.bx}, ${x.by})` : `(${x.rel.dx}, ${x.rel.dy})`}
              </button>
            ))}
          </div>
          <label className="check" style={{ marginBottom: '0.6rem' }}>
            <input type="checkbox" checked={abs} onChange={(ev) => setAbs(ev.target.checked)} /> absolute coordinates for block ({bx}, {by})
          </label>
          <table className="threads" aria-label="Counts">
            <tbody>
              <tr><td style={WRAP}>this block ({plural(walk.points.length, 'thread')})</td><td style={WRAP} data-readout="block">{thousands(walk.tested)} tests ({thousands(walk.dummyTests)} on dummies), {thousands(walk.accepted)} accepted ({pct(walk.tested ? walk.accepted / walk.tested : 0)})</td></tr>
              <tr><td style={WRAP}>thread at point ({pi}, {pj})</td><td style={WRAP} data-readout="thread">{threadStats.scanned} scanned ({threadStats.dummy} dummies), {threadStats.accepted} accepted</td></tr>
              <tr><td style={WRAP}>whole grid, direct</td><td style={WRAP} data-readout="direct">{thousands(points)} points × {thousands(world.atoms.length)} atoms = {thousands(grid.directPairs)} pairs</td></tr>
              <tr><td style={WRAP}>whole grid, binned</td><td style={WRAP} data-readout="binned">{thousands(grid.tests)} tests ({pct(grid.tests / grid.directPairs)} of direct), {thousands(grid.realTests)} on real atoms, {thousands(grid.accepted)} accepted</td></tr>
              {prev ? (
                <tr data-readout="doubled"><td style={WRAP}>against the box before (same capacity)</td><td style={WRAP}>points ×{(points / prev.points).toFixed(0)}, atoms ×{(world.atoms.length / prev.atoms).toFixed(0)}, direct pairs ×{(grid.directPairs / prev.direct).toFixed(0)}, binned tests ×{(grid.tests / prev.tests).toFixed(1)} (tests per point ×{(stats.perPoint / prev.perPoint).toFixed(2)})</td></tr>
              ) : null}
              <tr><td style={WRAP}>far field dropped here</td><td style={WRAP} data-readout="far">kept {nf.near.toFixed(3)} from {nf.nearAtoms} atoms; dropped {nf.far.toFixed(3)}; direct {nf.direct.toFixed(3)}</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="cols" style={{ marginTop: '0.9rem' }}>
        <div className="panel" data-mask="">
          <b className="ui">One staged atom, across the block's warps</b>
          <p className="ui muted" style={{ margin: '0.3rem 0 0.4rem', fontSize: '0.8rem' }}>
            Entry ({e.rel.dx}, {e.rel.dy}) of the list, bin ({e.bx}, {e.by}). Pick a slot: every thread tests the atom in it
            against the cutoff around its own point.
          </p>
          <div className="seg" role="group" aria-label="Slot" style={{ marginBottom: '0.5rem', flexWrap: 'wrap' }}>
            {e.slots.map((s, k) => (
              <button key={k} aria-pressed={k === sl} onClick={() => setPick({ entry: e.k, slot: k })} style={{ fontSize: '0.78rem' }}>
                {s < 0 ? `slot ${k}: dummy` : `slot ${k}: atom ${s}`}
              </button>
            ))}
          </div>
          {masks.map((mk, w) => (
            <div key={w} className="row" style={{ gap: 2, marginBottom: 4 }} data-warp={w} data-diverges={diverges(mk) ? 'yes' : 'no'}>
              <span className="muted" style={{ width: '3.6rem', fontSize: '0.78rem' }}>warp {w}</span>
              {mk.map((on, l) => (
                <svg key={l} width="11" height="14" aria-hidden="true">
                  <rect x="0.5" y="0.5" width="10" height="13" rx="1.5"
                    fill={e.slots[sl] < 0 ? `url(#${hatch})` : on ? 'var(--ok)' : 'var(--cell)'} stroke={on ? 'var(--ok)' : 'var(--rule)'} />
                </svg>
              ))}
              <span style={{ marginLeft: 6 }}>
                {e.slots[sl] < 0 ? <span className="chip">dummy: tested, adds 0</span>
                  : diverges(mk) ? <span className="chip fail">diverges: {mk.filter(Boolean).length} accept, {mk.length - mk.filter(Boolean).length} reject</span>
                  : <span className="chip ok">no divergence: all {mk.some(Boolean) ? 'accept' : 'reject'}</span>}
              </span>
            </div>
          ))}
        </div>
        <div className="panel" data-capacity-panel="">
          <b className="ui">Bin capacity</b>
          <div className="row" style={{ margin: '0.3rem 0' }}>
            <Field label={`Capacity ${plural(capacity, 'atom')} (largest bin holds ${maxOcc})`}>
              <input type="range" min={1} max={Math.max(1, maxOcc)} value={capacity} aria-label="bin capacity" onChange={(ev) => setCap(+ev.target.value)} />
            </Field>
          </div>
          <p className="ui" style={{ margin: 0 }} data-readout="capacity">
            Dummy slots: {thousands(binning.dummy)} of {thousands(bins * bins * capacity)} ({pct(binning.dummy / (bins * bins * capacity))}).
            Overflow: {plural(binning.overflow.length, 'atom')} ({pct(binning.overflow.length / Math.max(1, world.atoms.length), 1)}), summed on the host
            afterwards: {binning.overflow.length <= 0.03 * world.atoms.length ? 'under' : 'over'} the book's 3 %.
          </p>
        </div>
      </div>

      <div className="panel" style={{ marginTop: '0.9rem' }} data-book3d={n3h.total}>
        <b className="ui">3D, the book's case (p. 525)</b>
        <div className="row" style={{ margin: '0.4rem 0' }}>
          <Field label="Cutoff">
            <select value={c3} onChange={(ev) => setC3(+ev.target.value)}>
              {CUT_3D.map((v) => <option key={v} value={v}>{v} Å</option>)}
            </select>
          </Field>
          <Field label="Bin side">
            <select value={b3} onChange={(ev) => setB3(+ev.target.value)}>
              {BIN_3D.map((v) => <option key={v} value={v}>{v} Å</option>)}
            </select>
          </Field>
        </div>
        <p className="ui" style={{ margin: 0 }} data-readout="book3d">
          Cutoff + half the bin diagonal: <b>{thousands(n3h.total)} bins</b> ({thousands(n3h.full)} fully inside). With Fig. 21.13's
          whole diagonal: {thousands(n3f.total)}. At uniform density about <b>{pct(share3)}</b> of the atoms a thread tests
          pass its cutoff test.
        </p>
      </div>
    </div>
  );
}

// ================================================================ the explorer

const VIEW_LABEL: Record<ExplorerView, string> = { mapping: 'Who writes where', writes: 'Write statements', cutoff: 'Cutoff and bins' };

export function PairsExplorer({
  title,
  task,
  views,
  initialKernel = 'scatter',
  foot,
}: {
  title: string;
  task?: ReactNode;
  views: ExplorerView[];
  initialKernel?: KernelId;
  foot?: ReactNode;
}) {
  const [view, setView] = useState<ExplorerView>(views[0]);
  return (
    <Bench title={title} task={task} foot={foot}>
      {views.length > 1 ? (
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <span className="seg" role="group" aria-label="View">
            {views.map((v) => <button key={v} aria-pressed={v === view} onClick={() => setView(v)}>{VIEW_LABEL[v]}</button>)}
          </span>
        </div>
      ) : null}
      {view === 'mapping' ? <MappingView initialKernel={initialKernel} /> : view === 'writes' ? <WritesView /> : <CutoffView />}
    </Bench>
  );
}
