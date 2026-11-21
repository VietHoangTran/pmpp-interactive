import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  CUDA_QUEUES,
  DEFAULTS,
  HALO_BOOK,
  HIP_QUEUES,
  LANE_NAME,
  MAX_CHUNKS,
  MAX_STREAMS,
  ONE_ENGINE,
  PRESETS,
  START,
  SNIPPET,
  bestChunks,
  haloLanes,
  parseLabRows,
  simulate,
  stepAt,
  stepOp,
  sweep,
  type ClosedForm,
  type LabParts,
  type LabStaged,
  type PipeConfig,
  type Schedule,
  type View,
} from '../../sim/chunkpipe';
import { PinnedScene, PipeLanes, fmtN, lanesHeight } from './PinnedScene';

const LAB = 'ch23/lab1_overlap';
const CHECKED = '2026-11-20';
const STATUS_CU = 'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware';
const STATUS_METAL = 'verified on Metal';

// ---------------------------------------------------------------- Fig. 23.13, redrawn
function HaloBench() {
  const [x, setX] = useState(HALO_BOOK.exchange);
  const t = { ...HALO_BOOK, exchange: x };
  const plain = haloLanes(t, false);
  const over = haloLanes(t, true);
  const span = Math.max(plain.total, 1);
  const W = 720;
  const L = 112;
  const X = (v: number) => L + (v / span) * (W - L - 44);
  const SHORT: Record<string, string> = {
    'halo exchange (top)': 'top halo', 'halo exchange (bottom)': 'bottom halo', 'reduce L2 norm': 'reduce',
  };
  const diagram = (h: ReturnType<typeof haloLanes>, y0: number, title: string, key: string) => {
    let gpuRow = 0;
    return (
      <g data-halo={key} data-total={h.total}>
        <text x={8} y={y0 - 8} fontSize="12" fontWeight="600">{title}</text>
        <text x={L - 8} y={y0 + 16} fontSize="11" textAnchor="end">GPU</text>
        <text x={L - 8} y={y0 + 62} fontSize="11" textAnchor="end">Network</text>
        {h.bars.map((b, i) => {
          const sub = b.lane === 'gpu' ? gpuRow++ : 0;
          const y = b.lane === 'gpu' ? y0 + (sub === 0 ? 0 : 12 + sub * 9) : y0 + 48;
          const hgt = b.lane === 'gpu' ? (sub === 0 ? 20 : 8) : 22;
          const wide = X(b.to) - X(b.from);
          const text = wide > b.label.length * 5.6 ? b.label : SHORT[b.label] ?? b.label;
          return (
            <g key={i} data-bar={b.label}>
              <rect x={X(b.from)} y={y} width={Math.max(1, wide)} height={hgt} rx="2" fill="var(--ink)"
                fillOpacity={b.label.startsWith('reduce') ? 0.85 : b.label.includes('top') ? 0.55 : b.label.includes('bottom') ? 0.35 : 0.7} />
              {hgt >= 20 && wide > text.length * 5.6 ? (
                <text x={(X(b.from) + X(b.to)) / 2} y={y + hgt / 2 + 4} fontSize="10" textAnchor="middle"
                  style={{ fill: b.label.includes('bottom') ? 'var(--ink)' : 'var(--page)' }}>{text}</text>
              ) : null}
              {sub === 1 ? (
                <text x={X(b.to) + 5} y={y + 13} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>top and bottom rows</text>
              ) : null}
            </g>
          );
        })}
        <text x={X(h.total) + 5} y={y0 + 64} fontSize="11" fontWeight="700">{fmtN(h.total)}</text>
      </g>
    );
  };
  return (
    <Bench
      title="Two modes, then one"
      task="Drag the exchange time up. At what point does the overlapped iteration start to grow?"
      foot="Our redrawing of Fig. 23.13 in the book's proportions: each boundary row takes a quarter of the interior's time, as the figure draws it. Unitless. Model: sim/chunkpipe.ts, haloLanes."
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}>
          <span>Each halo exchange takes <b data-out="halo-x">{fmtN(x)}</b></span>
          <input type="range" min="0.5" max="8" step="0.5" value={x} aria-label="Time of each halo exchange" onChange={(e) => setX(Number(e.target.value))} />
        </label>
        <span className="muted">computing the interior takes {HALO_BOOK.internal}, each boundary row {HALO_BOOK.boundary}, the norm reduction {HALO_BOOK.reduce}</span>
      </div>
      <div className="panel" style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${W} 220`} role="img" aria-label="One stencil iteration without and with overlap" style={{ width: '100%', minWidth: 560, display: 'block' }}>
          {diagram(plain, 26, 'Without overlap: compute everything, then communicate', 'plain')}
          {diagram(over, 134, 'With overlap: boundary rows first, exchange while the interior computes', 'overlap')}
        </svg>
      </div>
      <p style={{ margin: '0.7rem 0 0' }} data-out="halo-verdict">
        Without overlap the iteration takes <b>{fmtN(plain.total)}</b>; with it, <b>{fmtN(over.total)}</b>.{' '}
        {over.hidden
          ? 'Both exchanges end before the interior does: the communication is hidden.'
          : `The exchanges run ${fmtN(over.exposed)} past the interior: that much communication is no longer hidden.`}{' '}
        The norm reduction waits for everything either way.
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the reader's device
interface DeviceData {
  backend: string;
  view: View;
  device: string;
  parts: LabParts;
  staged: LabStaged[];
}
function useDevice(): DeviceData[] {
  const { reports } = useReports();
  return realBackends(reports).flatMap((b) => {
    const view: View = b === 'metal' ? 'metal' : 'cuda';
    const rows = benchRows(reports[b], LAB) as unknown as Record<string, unknown>[];
    const got = parseLabRows(rows, view, b);
    return got.parts ? [{ backend: b, view, device: String(reports[b].device?.name ?? b), parts: got.parts, staged: got.staged }] : [];
  });
}
const deviceConfig = (d: DeviceData): PipeConfig => ({
  ...DEFAULTS, a: d.parts.a, b: d.parts.b, c: d.parts.c, view: d.view, engines: d.parts.engines ?? 2, n: 16, streams: 4,
});
const sameParts = (cfg: PipeConfig, d: DeviceData) =>
  cfg.view === d.view && Math.abs(cfg.a - d.parts.a) < 1e-9 && Math.abs(cfg.b - d.parts.b) < 1e-9 && Math.abs(cfg.c - d.parts.c) < 1e-9;

function DeviceCompare({ cfg, d }: { cfg: PipeConfig; d: DeviceData }) {
  const ns = [1, 2, 4, 8, 16, 32, 64];
  const model = sweep(cfg, ns);
  const points = d.staged.filter((r) => r.streams === cfg.streams && (d.view === 'metal' || (r.order === cfg.order && r.host === cfg.host)));
  const W = 520;
  const H = 190;
  const pad = { l: 46, r: 14, t: 14, b: 34 };
  const ymax = Math.max(...model.map((m) => m.total), ...points.map((p) => p.units), 1) * 1.08;
  const X = (n: number) => pad.l + (Math.log2(n) / 6) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - (v / ymax) * (H - pad.t - pad.b);
  const k = d.view === 'metal' ? 'K' : 'S';
  return (
    <div className="panel" data-device={d.backend} style={{ marginTop: '0.8rem' }}>
      <p style={{ margin: '0 0 0.4rem' }}>
        <b>{d.device}</b> ({d.backend}): lab 23.1's measured parts set the totals, the largest scaled to 24 units (1 unit ={' '}
        {fmtN(d.parts.unit * 1000)} ms). Line: the model at {k} = {cfg.streams}
        {d.view === 'metal' ? '' : `, ${cfg.order === 'depth' ? 'chunk by chunk' : 'all copy-ins first'}, ${cfg.host}`}. Dots: what the
        lab measured with the same settings.
      </p>
      {points.length ? (
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Model and measured totals against the number of chunks" style={{ width: '100%', maxWidth: W, display: 'block' }}>
          <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke="var(--rule)" />
          <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} stroke="var(--rule)" />
          {ns.map((n) => (
            <text key={n} x={X(n)} y={H - pad.b + 14} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{n}</text>
          ))}
          <text x={(W + pad.l) / 2} y={H - 4} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>chunks</text>
          <text x={8} y={pad.t - 2} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>units</text>
          {[0, 0.5, 1].map((f) => {
            const v = Math.round(ymax * f * 0.92);
            return (
              <g key={f}>
                <line x1={pad.l - 4} y1={Y(v)} x2={pad.l} y2={Y(v)} stroke="var(--ink-3)" />
                <text x={pad.l - 7} y={Y(v) + 3.5} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{v}</text>
              </g>
            );
          })}
          <polyline points={model.map((m) => `${X(m.n)},${Y(m.total)}`).join(' ')} fill="none" stroke="var(--ink)" strokeWidth="1.8" />
          {points.map((p, i) => (
            <circle key={i} data-measured={p.chunks} data-units={p.units} cx={X(p.chunks)} cy={Y(p.units)} r="4.5" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.8" />
          ))}
        </svg>
      ) : (
        <p className="muted" style={{ margin: 0 }} data-device-empty>
          The lab measured {k} = {[...new Set(d.staged.map((r) => r.streams))].sort((x, y) => x - y).join(', ')}
          {d.view === 'metal' ? '' : ' (pinned, both orders; and one pageable row)'}. Pick one of those to see its measurements here.
        </p>
      )}
      <p className="muted" style={{ margin: '0.4rem 0 0' }}>
        Measurements from your machine, not expected values. Each is a whole call, its setup included; the model has no such
        cost unless you set one.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- the stepper
function Range({ label, value, min, max, step, onChange, out, disabled, aria }: {
  label: ReactNode; value: number; min: number; max: number; step: number; onChange: (v: number) => void; out: string; disabled?: boolean; aria: string;
}) {
  return (
    <label className="field" style={{ minWidth: '9.5rem' }}>
      <span>{label} <b data-out={out}>{fmtN(value)}</b></span>
      <input type="range" min={min} max={max} step={step} value={value} disabled={disabled} aria-label={aria} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}
function Seg<T extends string | number>({ label, value, options, onChange, disabled, name }: {
  label: string; value: T; options: { v: T; label: string }[]; onChange: (v: T) => void; disabled?: boolean; name: string;
}) {
  return (
    <span className="row" style={{ gap: '0.45rem' }}>
      <span className="muted">{label}</span>
      <span className="seg" role="group" aria-label={label} data-seg={name}>
        {options.map((o) => (
          <button key={String(o.v)} aria-pressed={value === o.v} disabled={disabled} onClick={() => onChange(o.v)} data-v={String(o.v)}>
            {o.label}
          </button>
        ))}
      </span>
    </span>
  );
}

function FormBox({ f, kind, total }: { f: ClosedForm; kind: 'e1' | 'nv'; total: number }) {
  return (
    <div style={{ marginTop: '0.5rem' }} data-form={kind} data-holds={f.holds ? 'yes' : 'no'} data-value={f.value}>
      <div>
        <b>{f.name}</b>: {f.formula} = <b data-out={kind + '-value'}>{fmtN(f.value)}</b>{' '}
        {f.holds ? <span className="chip ok">its conditions hold</span> : <span className="chip">conditions not met</span>}
      </div>
      <div className="muted" style={{ fontSize: '0.76rem' }}>
        {f.conditions.map((c) => (c.holds ? '✓ ' : '✗ ') + c.text).join('; ')}.{' '}
        {f.holds ? 'The schedule agrees.' : f.matches ? 'It happens to agree here anyway.' : `The schedule gives ${fmtN(total)} instead.`}
      </div>
    </div>
  );
}

function Stepper({ cfg, setCfg, devices }: { cfg: PipeConfig; setCfg: (c: PipeConfig) => void; devices: DeviceData[] }) {
  const cu = useLabFile(SNIPPET.cuda);
  const mm = useLabFile(SNIPPET.metal);
  const s: Schedule = useMemo(() => simulate(cfg), [cfg]);
  const best = useMemo(() => bestChunks(cfg), [cfg]);
  const [t, setT] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const metal = cfg.view === 'metal';
  const span = Math.max(s.total, s.serial, 1e-6);
  const k = t == null ? null : stepAt(s, t);
  const step = k == null ? null : s.steps[k];
  const curOp = step ? stepOp(step) : null;
  const op = curOp != null ? s.ops[curOp] : null;
  const atStep = step != null && t != null && Math.abs(step.t - t) < 1e-9;

  useEffect(() => {
    setT(null);
    setPlaying(false);
  }, [cfg]);
  const next = () => {
    if (t == null) return setT(s.steps.length ? s.steps[0].t : 0);
    const nx = s.steps.find((x) => x.t > t + 1e-9);
    setT(nx ? nx.t : s.total);
  };
  const back = () => {
    if (t == null) return;
    const prev = [...s.steps].reverse().find((x) => x.t < t - 1e-9);
    setT(prev ? prev.t : null);
  };
  const finished = t != null && t >= s.total - 1e-9 && (k == null || k === s.steps.length - 1);
  useEffect(() => {
    if (!playing) return;
    if (finished) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(next, 650);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, t, finished]);

  const set = (o: Partial<PipeConfig>) => setCfg({ ...cfg, ...o });
  const narration: ReactNode = t == null ? (
    <>Nothing has run yet. {s.ops.length} operations wait to be queued{metal ? ' by the CPU loop' : ' by the host thread'}. Press Step.</>
  ) : step ? (
    <>
      <b>t = {fmtN(t)}{atStep ? '' : ` (the last event was at ${fmtN(step.t)})`}.</b>{' '}
      {step.events.map((e, i) => (
        <span key={i} data-ev={e.kind}>
          {e.kind === 'wait' || e.kind === 'block' ? <i>{e.text}</i> : e.text}{' '}
        </span>
      ))}
      {finished ? <b> All done at {fmtN(s.total)}.</b> : null}
    </>
  ) : (
    <>t = {fmtN(t)}. Nothing has happened yet.</>
  );

  // The code beside the timeline: the part of the snippet that issues work.
  const slice = (src: string | null, from: number, to: number) => (src ? src.split('\n').slice(from - 1, to).join('\n') : '');
  const hl = op ? op.lines : [];
  const codeTabs = metal
    ? [{ label: 'Metal host', code: slice(mm, 21, 46), status: STATUS_METAL, highlight: hl }]
    : [{ label: 'CUDA', code: slice(cu, 34, 59), status: 'emulator-checked, front-end-checked', highlight: hl }];

  const devicesHere = devices.filter((d) => d.view === cfg.view || devices.length === 1);
  const shownDevice = devices.find((d) => sameParts(cfg, d));
  const H = lanesHeight(s) + 24;
  const scroller = useRef<HTMLDivElement>(null);

  return (
    <Bench
      title="Copy, compute, copy back"
      task="Step through the starting case first. Then find a setting that beats it, and one that loses all the overlap."
      foot={
        <>
          Each box is one operation, labelled with its chunk and shaded by its stream; the outlined box is the one that just
          started, and boxes of the same stream have a firmer outline. Hatched: the host blocked or waiting. Dotted: idle time of
          the busiest lane (the bubble). Times are unitless. Model: sim/chunkpipe.ts.
        </>
      }
    >
      <div data-stepper data-total={s.total} data-view={cfg.view}>
        <div className="row" style={{ marginBottom: '0.6rem', alignItems: 'flex-end' }}>
          <Range label="Chunks n" value={cfg.n} min={1} max={MAX_CHUNKS} step={1} out="n" aria="Number of chunks" onChange={(v) => set({ n: v })} />
          <Range label={metal ? 'Buffers in flight K' : 'Streams S'} value={cfg.streams} min={1} max={MAX_STREAMS} step={1} out="streams" aria={metal ? 'Buffers in flight' : 'Number of streams'} onChange={(v) => set({ streams: v })} />
          <Range label={metal ? 'Fill (copy-in)' : 'Copy-in'} value={cfg.a} min={0} max={32} step={0.5} out="a" aria="Total copy-in time" onChange={(v) => set({ a: v })} />
          <Range label="Kernel" value={cfg.b} min={0} max={32} step={0.5} out="b" aria="Total kernel time" onChange={(v) => set({ b: v })} />
          <Range label={metal ? 'Drain (copy-out)' : 'Copy-out'} value={cfg.c} min={0} max={32} step={0.5} out="c" aria="Total copy-out time" onChange={(v) => set({ c: v })} />
          <span className="muted" data-out="chunk-size">each chunk {cfg.items == null ? `1/${cfg.n}` : 'about 1/' + cfg.n} of the data; chunk j goes to {metal ? 'buffer' : 'stream'} j mod {cfg.streams}</span>
        </div>
        <div className="row" style={{ marginBottom: '0.6rem' }}>
          <Seg name="view" label="View" value={cfg.view} onChange={(v) => set({ view: v })} options={[{ v: 'cuda', label: 'CUDA / HIP' }, { v: 'metal', label: 'Metal' }]} />
          <Seg name="engines" label="Copy engines" value={cfg.engines} disabled={metal} onChange={(v) => set({ engines: v })}
            options={[{ v: 2, label: 'two' }, { v: 1, label: 'one, shared' }]} />
          <Seg name="order" label="Issue order" value={cfg.order} disabled={metal} onChange={(v) => set({ order: v })}
            options={[{ v: 'depth', label: 'chunk by chunk' }, { v: 'breadth', label: 'all copy-ins first' }]} />
          <Seg name="host" label="Host memory" value={cfg.host} disabled={metal} onChange={(v) => set({ host: v })}
            options={[{ v: 'pinned', label: 'pinned' }, { v: 'pageable', label: 'pageable' }]} />
          <Seg name="queues" label="Hardware queues" value={isFinite(cfg.queues) ? String(cfg.queues) : 'inf'} disabled={metal}
            onChange={(v) => set({ queues: v === 'inf' ? Infinity : Number(v) })}
            options={[{ v: 'inf', label: 'unlimited' }, { v: String(CUDA_QUEUES), label: `${CUDA_QUEUES} (CUDA)` }, { v: String(HIP_QUEUES), label: `${HIP_QUEUES} (HIP)` }]} />
          <Range label={<>Per-operation cost <span className="muted">(no vendor publishes this figure)</span></>} value={cfg.overhead} min={0} max={2} step={0.05} out="overhead" aria="Per-operation cost" onChange={(v) => set({ overhead: v })} />
        </div>
        {metal ? (
          <p className="muted" style={{ margin: '0 0 0.6rem' }}>
            Metal view: no copy engine, issue order, pinned memory or hardware queue applies. The CPU drains a buffer, refills it
            and commits; the GPU runs command buffers in commit order.
          </p>
        ) : null}
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <span className="muted">Presets:</span>
          {PRESETS.map((p) => (
            <button key={p.id} className="quiet" data-preset={p.id} onClick={() => setCfg({ ...p.cfg, view: cfg.view })}>{p.label}</button>
          ))}
          {devicesHere.map((d) => (
            <button key={'dev-' + d.backend} className="quiet" data-preset={'device-' + d.backend} onClick={() => setCfg(deviceConfig(d))}>
              your device (lab 23.1{devices.length > 1 ? `, ${d.backend}` : ''})
            </button>
          ))}
        </div>

        <div className="row" style={{ marginBottom: '0.6rem' }}>
          <button onClick={() => { setPlaying(false); setT(null); }} disabled={t == null}>Restart</button>
          <button onClick={() => { setPlaying(false); back(); }} disabled={t == null}>Back</button>
          <button className="primary" onClick={() => { setPlaying(false); next(); }} disabled={finished}>Step</button>
          <button onClick={() => setPlaying(!playing)} disabled={finished}>{playing ? 'Pause' : 'Play'}</button>
          <button onClick={() => { setPlaying(false); setT(s.total); }} disabled={finished}>Run to the end</button>
          <input type="range" min={0} max={1000} step={1} value={t == null ? 0 : Math.round((t / span) * 1000)} aria-label="Time"
            style={{ flex: '1 1 10rem' }} onChange={(e) => { setPlaying(false); setT((Number(e.target.value) / 1000) * span); }} />
          <span className="muted" aria-live="polite" data-out="clock">t = {t == null ? '·' : fmtN(t)} of {fmtN(s.total)}</span>
        </div>

        <div className="narrate" role="status" data-narrate style={{ marginBottom: '0.8rem' }}>{narration}</div>

        <div className="panel" style={{ overflowX: 'auto', marginBottom: '0.8rem', padding: '0.5rem' }} ref={scroller}>
          <svg viewBox={`0 0 940 ${H}`} role="img" aria-label="Timeline of the chunked pipeline, one lane per resource"
            style={{ width: '100%', minWidth: 640, display: 'block' }} data-timeline>
            <PipeLanes s={s} t={t ?? -1} x={0} y={18} w={940} span={span} current={curOp} id="cp-stepper" />
          </svg>
        </div>

        <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 7fr) minmax(0, 5fr)' }}>
          <div>
            {(metal ? mm : cu) ? (
              <Code tabs={codeTabs} firstLine={metal ? 21 : 34}
                caption={<>The part of <code>labs/{metal ? SNIPPET.metal : SNIPPET.cuda}</code> that queues the work; the marked lines issued the operation that just started.</>} />
            ) : null}
          </div>
          <div className="panel" data-readouts>
            <table className="threads wrap">
              <tbody>
                {([
                  ['Total', <b>{fmtN(s.total)}</b>, 'total'],
                  [`In one piece, a + b + c${cfg.overhead > 0 ? ' + costs' : ''}`, fmtN(s.serial), 'serial'],
                  ['Speed-up', `${fmtN(s.speedup)}×`, 'speedup'],
                  ['Busiest lane', `${LANE_NAME[s.bottleneck]} (${fmtN(s.busy[s.bottleneck] ?? 0)} busy)`, 'bottleneck'],
                  ['Its idle time, the bubble', fmtN(s.bubble), 'bubble'],
                  ['Fastest chunk count, 1 to 64', best.n === 1 ? `n = 1 (${fmtN(best.total)}): cutting does not help here` : `n = ${best.n} (${fmtN(best.total)})`, 'best-n'],
                ] as [string, ReactNode, string][]).map(([k, v, id]) => (
                  <tr key={id}>
                    <td>{k}</td>
                    <td data-out={id}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {s.e1 ? <FormBox f={s.e1} kind="e1" total={s.total} /> : null}
            {s.nvidia && cfg.c === 0 ? <FormBox f={s.nvidia} kind="nv" total={s.total} /> : null}
            {s.lowerBound != null ? (
              <p style={{ margin: '0.5rem 0 0' }} data-out="lower-bound">
                No closed form here. The CPU does every fill and drain and the GPU every kernel, so the total is at least n ·
                max(fill + drain, kernel) per chunk = <b>{fmtN(s.lowerBound)}</b>.
              </p>
            ) : null}
          </div>
        </div>
        {shownDevice ? <DeviceCompare cfg={cfg} d={shownDevice} /> : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
/** q-twostreams: three equal stages, 12 chunks, from the stepper's model. */
export const TWO_STREAMS = (streams: number) => simulate({ ...DEFAULTS, a: 12, b: 12, c: 12, n: 12, streams });
const X1 = (v: number) => v.toFixed(1);
/** The false dependency: the "equal stages" preset with 16 chunks in 16 streams, breadth first. */
export const FALSE_DEP: PipeConfig = { ...PRESETS.find((p) => p.id === 'equal')!.cfg, n: 16, streams: 16, order: 'breadth', queues: CUDA_QUEUES };
export const FALSE_DEP_FREE = simulate({ ...FALSE_DEP, queues: Infinity }).total;
export const FALSE_DEP_8 = simulate(FALSE_DEP).total;

export const questionsS233: Record<string, Question> = {
  'ch23/23.3/q-onestream': {
    kind: 'choice',
    id: 'ch23/23.3/q-onestream',
    prompt: <>You split a job into 8 chunks but queue all of them in one stream, on a GPU with two copy engines. What speed-up over the unsplit job should you expect?</>,
    options: [
      { label: "About 1×, or a little worse once each chunk's fixed cost counts.", correct: true },
      { label: 'About 3×: one lane each for the copy-in, the kernel and the copy-out.', why: 'That is what three busy lanes could give; one stream never lets two of its operations run at once.' },
      { label: 'About 8×: each chunk is an eighth of the work, so each finishes sooner.', why: 'Cutting the work does not make the whole faster. Only overlap can, and one stream has none.' },
      { label: "It depends on the kernel's occupancy, which sets how much copying fits beside it.", why: 'Occupancy decides how fast each kernel runs, not whether a copy can run under it. In one stream it cannot.' },
    ],
    explain: (
      <p>
        A stream runs its operations one after another, in the order they were inserted (pp. 555–556). Chunk 1's copy-in waits
        for chunk 0's copy-out, so only one operation ever runs. In the stepper, set S to 1 and move n anywhere: the total
        stays at the one-piece time, and grows with a per-operation cost.
      </p>
    ),
    source: 'ch. 23 §23.3, pp. 555–556; the stepper (sim/chunkpipe.ts)',
  },
  'ch23/23.3/q-pageable': {
    kind: 'choice',
    id: 'ch23/23.3/q-pageable',
    prompt: (
      <>
        Your staged loop is no faster than the serial one, and a trace shows each <code>cudaMemcpyAsync</code> returning only
        after its copy has finished. Cause and fix?
      </>
    ),
    options: [
      {
        label: <>The host buffers came from <code>malloc</code>, so each copy ran synchronously. Allocate them with <code>cudaMallocHost</code>.</>,
        correct: true,
      },
      { label: <>The streams were created with <code>cudaStreamNonBlocking</code>, which makes each of their copies wait for the host.</>, why: 'That flag only stops a stream from synchronising with the legacy default stream. It helps overlap, if anything.' },
      { label: <><code>cudaMemcpyAsync</code> needs <code>cudaMemcpyDefault</code> as its direction, or it falls back to a synchronous copy.</>, why: 'The direction argument has nothing to do with when the call returns.' },
      { label: 'The kernel launch needs a fourth parameter of 1, or the copies around it go to the default stream.', why: 'The fourth parameter is a stream, not a number, and the symptom is in the copies.' },
    ],
    explain: (
      <p>
        From pageable memory the runtime stages the copy through a pinned buffer, and the host has to take part (pp. 562–563).
        NVIDIA's reference: such a copy "will revert to a synchronous behavior which will not overlap with other work" (A6).
        AMD's header: "the memory copy will be performed synchronously" (B4); there the pinned allocation is{' '}
        <code>hipHostMalloc</code>. The stepper's pageable switch shows the result.
      </p>
    ),
    source: 'ch. 23 §23.3, pp. 562–563; notes/today/ch23-facts.md A6, B4',
  },
  'ch23/23.3/q-bottleneck': {
    kind: 'numeric',
    id: 'ch23/23.3/q-bottleneck',
    prompt: (
      <>
        A job's totals: copy-in 8 ms, kernel 12 ms, copy-out 4 ms. You cut it into 4 equal chunks, one stream per chunk, on a
        GPU with two copy engines, and the per-chunk cost is negligible. How long does it take, in ms?
      </>
    ),
    answer: 15,
    unit: 'ms',
    explain: (
      <p>
        Per chunk: 2, 3 and 1 ms. The first chunk needs all three, 6 ms. After that the kernel lane, the slowest, finishes one
        more chunk every 3 ms: 6 + 3 × 3 = 15 ms, against 24 ms in one piece. That is a + b + c + (n − 1)·max(a, b, c) for
        per-chunk times. The numbers are made up for the exercise; your device's come from lab 23.1.
      </p>
    ),
    source: 'notes/today/ch23-facts.md E1 (our arithmetic, after Best Practices 10.1.2)',
  },
  'ch23/23.3/q-twostreams': {
    kind: 'choice',
    id: 'ch23/23.3/q-twostreams',
    prompt: (
      <>
        Three equal stages, 12 chunks. The stepper gives about {X1(TWO_STREAMS(2).speedup)}× with 2 streams and about{' '}
        {X1(TWO_STREAMS(3).speedup)}× with 3, the most any stream count gives here. Why is a second stream not enough to keep
        all three lanes busy?
      </>
    ),
    options: [
      {
        label: "Chunk i + 2 shares a stream with chunk i, so its copy-in waits for chunk i's copy-out.",
        correct: true,
      },
      { label: 'The GPU runs at most two streams at once, so a third stream would wait for a free slot.', why: 'Nothing limits it to two. The hardware-queue limits are 8 and 4, and they bind only with more streams.' },
      { label: "The second stream shares the first one's copy engine, so their copy-ins take turns.", why: 'Every stream shares the engines; with two engines, each direction has its own.' },
      { label: 'A stream can hold only one kernel at a time, so each kernel waits for the one before.', why: 'A stream holds any number of operations, in order.' },
    ],
    explain: (
      <p>
        A stream runs in order. With 2 streams, chunk 2 sits in stream 0 behind chunk 0's copy-out, so it cannot start until
        chunk 0 is all the way through; keeping three stages busy takes three chunks in flight, one per stream. Each pair of chunks then takes 3 units: 19 for 12 chunks, against 36 in one piece and 14 with 3 streams. The
        rule of thumb we use: S × the slowest per-chunk stage must cover a whole chunk's a + b + c, here 3 × 1 ≥ 3.
      </p>
    ),
    source: 'our model (sim/chunkpipe.ts tests: n = 12 with 1, 1, 1 per chunk gives 19 with 2 streams, 14 with 3)',
  },
  'ch23/23.3/q-overhead': {
    kind: 'choice',
    id: 'ch23/23.3/q-overhead',
    prompt: <>Going from 16 to 256 chunks made your staged job slower. What is the most likely reason?</>,
    options: [
      {
        label: 'Each chunk adds a fixed cost to every copy and kernel, and the bubble it shrinks was small.',
        correct: true,
      },
      { label: 'Smaller copies reach a higher share of the link, which starves the kernels of memory bandwidth.', why: 'Nothing in the documentation checked here says so, and the model slows down without any such effect.' },
      { label: "256 chunks exceed a stream's documented limit of 64 queued operations, so the host waits for room.", why: 'No such limit appears in the documentation checked for this chapter. The 64 you may remember is Metal\'s default for uncompleted command buffers on one queue (C1).' },
      { label: 'Smaller kernels have higher occupancy, which slows the copies running beside them.', why: 'A small grid may not even fill the GPU; occupancy does not rise because a kernel is shorter.' },
    ],
    explain: (
      <p>
        With fixed totals the bubble shrinks as 1/n, while the fixed cost per operation grows as n (E3: no vendor publishes
        it). In the stepper, set a cost and watch the "fastest chunk count" read-out. A second cause can add to it: CUDA maps
        streams onto 8 work queues by default, and when unrelated streams share one, "a false dependency is created" (A8).
      </p>
    ),
    source: 'notes/today/ch23-facts.md E3, A8; the stepper',
  },
  'ch23/23.3/q-hip-engines': {
    kind: 'choice',
    id: 'ch23/23.3/q-hip-engines',
    prompt: <>On ROCm 10.2 your program reads <code>asyncEngineCount</code> = 8. What does that tell you about copies in both directions overlapping?</>,
    options: [
      { label: 'Nothing: ROCm sets it to a runtime constant, 8, on every device. Measure instead.', correct: true },
      { label: 'Eight copies can run at once, four in each direction, if they are in different streams.', why: 'The value is the same on every AMD device, whatever its hardware.' },
      { label: 'Copies overlap in both directions, as they do on CUDA when the field reads 2.', why: "That reading is CUDA's (A7). On ROCm the field is a constant (B3)." },
      { label: 'The GPU has eight DMA engines, and each stream gets one of them for its copies.', why: 'Nothing in the field says so; it is set from a constant in the runtime.' },
    ],
    explain: (
      <p>
        In the ROCm runtime the property is set from <code>kMaxAsyncQueues</code>, a constant 8 (B3). AMD's guide says only that
        a value above zero means the device supports concurrent transfers. Lab 23.1 measures whether your copies overlap.
      </p>
    ),
    source: 'notes/today/ch23-facts.md B3',
  },
  'ch23/23.3/q-metal': {
    kind: 'choice',
    id: 'ch23/23.3/q-metal',
    prompt: <>Porting the staged loop to Metal on Apple silicon, what plays the role of the copy-in?</>,
    options: [
      {
        label: 'The CPU writing chunk i + 1 into one of K shared buffers while the GPU computes chunk i from another.',
        correct: true,
      },
      { label: 'A blit pass on a dedicated copy engine, which runs beside the compute pass the way a CUDA copy-in does.', why: 'Apple documents a blit as a kind of pass, not as separate copy hardware (C4), and with shared memory there is nothing to copy.' },
      { label: 'cudaMemcpyAsync, through a compatibility layer that maps CUDA calls onto Metal.', why: 'There is no such layer.' },
      { label: 'A private buffer that the GPU fills from host memory before each compute pass.', why: 'Something still has to put the input there. A shared buffer the CPU writes needs no copy at all (C4).' },
    ],
    explain: (
      <p>
        A shared buffer is the GPU's memory (C4). Apple's documented overlap is several buffer instances in flight, with the rule
        to "finalize all CPU writes to one buffer instance before you commit a command buffer that references it" (C5). With K
        = 1 nothing overlaps, and writing into a buffer whose command buffer has not completed is a race. In the stepper's
        Metal view, set K to 1, then 2.
      </p>
    ),
    source: 'notes/today/ch23-facts.md C4, C5',
  },
};

// ---------------------------------------------------------------- the page
export function S233() {
  const cu = useLabFile(SNIPPET.cuda);
  const mm = useLabFile(SNIPPET.metal);
  const metalKernel = useLabFile(SNIPPET.metalKernel);
  const devices = useDevice();
  const [cfg, setCfg] = useState<PipeConfig>(START);
  const stepperRef = useRef<HTMLDivElement>(null);
  const load = (c: PipeConfig) => {
    setCfg(c);
    stepperRef.current?.scrollIntoView({ block: 'start' });
  };
  const pPipe = PRESETS.find((p) => p.id === 'p-pipeline')!.cfg;
  const sPipe = simulate(pPipe);
  const sOne = simulate(ONE_ENGINE);
  const sOneB = simulate({ ...ONE_ENGINE, order: 'breadth' });

  return (
    <>
      <p>
        The loop of section 23.2 has a flaw the book points out at once: at any moment only part of the machine works. While
        every GPU computes, the network is idle; while the GPUs exchange halo rows, the GPUs sit mostly idle. This page takes
        the book's cure and the three tools it needs, streams, asynchronous copies and pinned memory, and applies them where
        you will use them first: on one GPU, hiding the copies between host and device behind a kernel.
      </p>
      <Source section="23.3" pages="553–554" />

      <h2>Two modes, each leaving half the hardware idle</h2>
      <p>
        The book's fix has two stages (Fig. 23.12). First, each GPU computes only its boundary rows, the ones its neighbours
        will read as halos. Then it does two things at once: it sends those rows, and it computes the rest of its partition.
        If the sending takes less time than the interior, the exchange costs nothing extra, which needs enough interior rows
        per GPU. One step cannot be overlapped: the reduction of the norm, which needs every point.
      </p>
      <Source section="23.3" pages="553–555" figs="Figs. 23.12, 23.13" />
      <Simplification>
        <p>
          In the explorer below, boundary rows are computed beside the interior without slowing it, as Fig. 23.13 draws
          them. The bottom exchange starts after the top one because <code>MPI_Sendrecv</code> blocks the host (p. 562).
        </p>
      </Simplification>
      <HaloBench />
      <p>
        Now shrink the picture to one GPU. A job that copies its input to the device, runs a kernel and copies the result back
        has the same flaw. During the kernel the copy hardware sits idle; during the copies the SMs do. The cure is the same
        too: cut the work into pieces, and move one piece while another computes. The rest of the page builds that, one tool
        at a time.
      </p>
      <Beyond>
        <p>
          The book overlaps network traffic. The one-GPU version, copies over the host-device link, is NVIDIA's (Best
          Practices Guide 10.1.2, ch23-facts A7); the model and the snippets here are ours.
        </p>
      </Beyond>

      <h2>Streams: ordered queues that may run side by side</h2>
      <p>
        A stream is a queue of GPU operations: kernel launches, copies and a few others. Inside one stream they run in the
        order you inserted them. Between two streams there is no order at all, and the GPU may run their work at the same time.
      </p>
      <Source section="23.3" pages="555–556" />
      <p>
        You choose the stream with the fourth launch parameter. The first two, grid and block, are chapter 2's; the third, the
        dynamic shared memory size, is chapter 5's; the fourth names the stream:{' '}
        <code>kernel&lt;&lt;&lt;grid, block, 0, stream&gt;&gt;&gt;(...)</code>. Leave it out and the launch goes into the
        default stream.
      </p>
      <Source section="23.3" pages="556" />
      <p>
        And the launch is asynchronous: when the call returns, the kernel has been inserted into its stream. It may not have
        finished, or even started; it runs once everything ahead of it in that stream is done. You saw this for one stream on{' '}
        <a href={href('ch02', '2.2')}>chapter 2's timeline</a>. With several streams, the host can queue work for all of them
        and get on with something else.
      </p>
      <Today checked={`${CHECKED} (ch23-facts A3, A9; ch22-facts A2)`}>
        <p>
          The legacy default stream "synchronizes with all other streams" except non-blocking ones. Avoid it with{' '}
          <code>cudaStreamNonBlocking</code> or <code>--default-stream per-thread</code>. Kernels overlap only on non-default
          streams, with no dependency and free resources.
        </p>
      </Today>
      <p>
        Chapter 22 sent you here for streams as a way to run small tasks side by side (<a href={href('ch22', '22.3')}>§22.3</a>).
        The rest of this page uses the same tool on copies.
      </p>
      <Ask q={questionsS233['ch23/23.3/q-onestream']} />

      <h2>Asynchronous copies need pinned memory</h2>
      <p>
        Copies have an asynchronous form too: <code>cudaMemcpyAsync</code> takes a stream and returns at once. There is a catch,
        and the book explains it from the bottom up.
      </p>
      <Source section="23.3" pages="560, 562–563" />
      <p>
        Your process sees a large, contiguous virtual address space, cut into pages. The operating system maps each page onto a
        frame of physical memory, and it may move a page, or write it out to disk, at any time: that is how it keeps processes
        apart and serves more memory than it has. Memory from <code>malloc</code> is like that. It is pageable.
      </p>
      <p>
        The copy hardware does not see virtual addresses. A DMA engine moves data by physical address, without the CPU. If the
        operating system moved a page in the middle of a copy, the engine would read or write a frame that now holds something
        else. So the runtime does not hand pageable memory to the engine. It copies in two steps instead: from your buffer into
        a staging buffer the operating system will not move, then by DMA to the device, and the reverse for the way back. The
        extra copy costs time, and since the CPU does it, the call cannot return until it is finished.
      </p>
      <p>
        Memory the operating system will not move is called pinned, or page-locked. <code>cudaMallocHost(&amp;h, bytes)</code>{' '}
        allocates it. The engine reads it directly, and the host thread is free as soon as the copy is queued.
      </p>
      <Source section="23.3" pages="562–563" />
      <Simplification>
        <p>
          The scene below stages the copy in pieces through two buffers and returns after the last piece. NVIDIA documents
          neither the buffers nor exactly when the call returns ("might be synchronous", A6).
        </p>
      </Simplification>
      <PinnedScene />
      <p>
        One correction to the book. It says <code>cudaMemcpyAsync</code> <em>requires</em> pinned memory, and NVIDIA's Best
        Practices Guide uses the same word. The runtime reference is more precise: from pageable memory the call still works,
        but it "will revert to a synchronous behavior which will not overlap with other work". The failure is silent: right
        results, no overlap.
      </p>
      <Today checked={`${CHECKED} (ch23-facts A6)`}>
        <p>
          On systems where the GPU uses the host's page tables (HMM or ATS), NVIDIA notes that <code>cudaMallocHost</code> "may
          not page-lock the allocated memory".
        </p>
      </Today>
      <p>
        <a href={href('appC', 'C.2')}>Appendix C, §C.2</a> takes pinned memory one step further: host memory the GPU reads in
        place, and what PyTorch's <code>pin_memory</code> does.
      </p>
      <Ask q={questionsS233['ch23/23.3/q-pageable']} />

      <h2>Cut the job into chunks</h2>
      <p>
        Now the pieces fit together. Split the data into n chunks. For each chunk, queue three operations in one stream: copy
        its input in, run the kernel on it, copy its result out. Spread the chunks over S streams. Inside a stream a chunk's
        three steps stay in order; across streams, one chunk's copy can run while another chunk's kernel does. Here is the loop
        written two ways, issuing each chunk's three operations in turn (depth first) or every copy-in first (breadth first),
        and the same job on Metal.
      </p>
      <Beyond>
        <p>
          The staged loop is NVIDIA's (Best Practices 10.1.2, A7); these files are ours. They split the data as lab 23.1 does,
          so what you read here is what you will measure there.
        </p>
      </Beyond>
      {cu && mm && metalKernel ? (
        <Code
          tabs={[
            { label: 'CUDA', code: cu, status: STATUS_CU },
            { label: 'Metal host', code: mm, status: STATUS_METAL },
            { label: 'Metal kernel', code: metalKernel, status: STATUS_METAL },
          ]}
          caption={
            <>
              From <code>labs/ch23/snippets/</code>. The course's HIP header maps every CUDA call in the CUDA file; no HIP
              build of it has run yet. The checks that give these statuses are in <code>labs/ch23/snippets/check.py</code>.
            </>
          }
        />
      ) : null}
      <Predict
        id="ch23/23.3/p-pipeline"
        prompt={
          <>
            8 equal chunks. For each chunk the copy-in takes 1, the kernel 2, the copy-out 1 (units, not milliseconds). One
            stream per chunk, two copy engines. Done in one piece it takes 32. How long does the pipelined job take?
          </>
        }
        unit="units"
      >
        {(v) => (
          <>
            <p>
              Load the case into the stepper and read its total. The model gives <b>{fmtN(sPipe.total)}</b>
              {v === sPipe.total ? ', as you said' : ''}: the first chunk needs 1 + 2 + 1 = 4 to get through, and after that
              the kernel lane, the slowest, finishes one more chunk every 2, so 4 + 7 × 2.
            </p>
            <button onClick={() => load(pPipe)} data-load="p-pipeline">Load it into the stepper</button>
          </>
        )}
      </Predict>
      <p>
        The stepper runs this loop on a model of the GPU's lanes. It starts on a case of its own, not the one you just
        predicted; step through it first.
      </p>
      <Simplification>
        <p>
          One kernel at a time. Each lane serves its operations in issue order, which NVIDIA does not document for one copy
          engine (E2). A pageable copy waits for its stream, then holds the host (the worst case allowed). Streams map
          round-robin onto in-order queues. No caches, no shared bandwidth. Unitless.
        </p>
      </Simplification>
      <div ref={stepperRef} style={{ scrollMarginTop: '1rem' }} className="wide">
        <Stepper cfg={cfg} setCfg={setCfg} devices={devices} />
      </div>
      <p>Things to find in the stepper:</p>
      <ol>
        <li>Set S to 1. However many chunks you cut, nothing overlaps. Why not?</li>
        <li>
          Load "equal stages" and try S = 2, then S = 3. Watch which copy-in waits, and what the narration says it waits for.
        </li>
        <li>Set a per-operation cost of 0.25 and move n from 1 to 64. Where is the fastest chunk count, and why does it exist?</li>
        <li>Switch the host memory to pageable. Which lane is busy now, and what happened to the overlap?</li>
      </ol>
      <p>
        The total the stepper reports is a + b + c + (n − 1)·max(a, b, c) for per-chunk times, as long as there is a copy engine
        per direction and enough streams: the first chunk passes through all three lanes, and then the slowest lane finishes one
        chunk per step. The idle time at the two ends, the bubble, is a + b + c − max(a, b, c) per chunk; smaller chunks shrink
        it. That closed form, and the condition that S times the slowest stage covers a whole chunk, are ours (ch23-facts
        E1), and the model's tests check them against the schedule.
      </p>
      <Today checked={`${CHECKED} (ch23-facts A7)`}>
        <p>
          NVIDIA prints this loop with a rough estimate: when the kernel time tE exceeds the transfer time tT, about tE +
          tT/nStreams staged, against tE + tT in one go. Its example copies one way only; the "Best Practices example" preset
          reproduces it.
        </p>
      </Today>
      <p>
        How many copies can move at once depends on copy engines. A GPU with one copy engine can run one asynchronous transfer
        alongside kernels; with two, one in each direction at the same time. CUDA reports the number as{' '}
        <code>asyncEngineCount</code>.
      </p>
      <Beyond>
        <p>Copy engines are not in the book; NVIDIA's Best Practices Guide 10.1.2 describes them (ch23-facts A7).</p>
      </Beyond>
      <Predict
        id="ch23/23.3/p-oneengine"
        prompt={
          <>
            Before you switch the stepper to one copy engine: 4 chunks of 1, 2, 1; one stream each; one copy engine shared by
            both directions; operations issued chunk by chunk. Done in one piece it takes 16. How long now?
          </>
        }
        unit="units"
      >
        {() => (
          <>
            <p>
              <b>{fmtN(sOne.total)}</b>: no gain at all. Chunk 0's copy-out sits ahead of chunk 1's copy-in in the one engine's
              queue, and it cannot start before kernel 0 has finished, so chunk 1's copy-in waits as well. Issue every copy-in
              first and the same work takes <b>{fmtN(sOneB.total)}</b>.
            </p>
            <div className="row">
              <button onClick={() => load(ONE_ENGINE)} data-load="one-depth">Load: chunk by chunk</button>
              <button onClick={() => load({ ...ONE_ENGINE, order: 'breadth' })} data-load="one-breadth">Load: all copy-ins first</button>
            </div>
          </>
        )}
      </Predict>
      <p>
        So with one engine the issue order decides the result. That rests on an assumption the model states: the engine serves
        copies in the order they were issued. NVIDIA documents no order for this case, so treat it as a hypothesis to test on a
        device whose <code>asyncEngineCount</code> is 1. (The one-engine cases are our arithmetic, ch23-facts E2.)
      </p>
      <p>
        Streams are not free either. CUDA maps them onto hardware work queues, 8 by default (<code>CUDA_DEVICE_MAX_CONNECTIONS</code>,
        1 to 32). When two unrelated streams land in one queue, "a false dependency is created which can lead to GPU work
        serialization". HIP uses at most 4 hardware queues per device by default and shares them round-robin. To see it,
        load "equal stages", set 16 chunks and 16 streams, issue all copy-ins first, and switch the queues from unlimited
        to 8. Which copy-in waits now, and for what?{' '}
        <button className="quiet" onClick={() => load(FALSE_DEP)} data-load="false-dep">Load that case</button>
      </p>
      <Today checked={`${CHECKED} (ch23-facts A8; ch22-facts B1)`}>
        <p>Neither limit is in the book. CUDA's is an environment variable; HIP's is <code>GPU_MAX_HW_QUEUES</code>.</p>
      </Today>
      <p>
        And each operation has a fixed cost, to set up a copy or launch a kernel, that no vendor publishes. Cut too fine and the
        costs outgrow the bubble you were shrinking. The stepper's cost is a dial for that reason, not a figure; lab 23.1 finds
        your device's best chunk count by measuring.
      </p>
      <Ask q={questionsS233['ch23/23.3/q-bottleneck']} label="Work it out" />
      <Ask q={questionsS233['ch23/23.3/q-twostreams']} />
      <Ask q={questionsS233['ch23/23.3/q-overhead']} />
      <p>
        <a href={href('ch23', 'labs')}>Lab 23.1, Hide the copies behind the kernel</a>, runs this loop on your GPU with pinned
        and pageable buffers, both issue orders and several stream counts. Once you have run it, the stepper offers your device's
        measured parts as a preset and plots what the lab measured beside the model.
      </p>

      <h2>On HIP and Metal</h2>
      <p>
        On HIP the same source goes through the course's header, which maps every call this page uses (unverified on AMD
        hardware so far). Two renames are not a prefix swap: <code>cudaMallocHost</code> becomes <code>hipHostMalloc</code>, and{' '}
        <code>cudaFreeHost</code> becomes <code>hipHostFree</code>. The trap is the same as on CUDA: AMD's header says
        that if a buffer is not pinned, "the memory copy will be performed synchronously". One difference to know before you
        measure: ROCm sets <code>asyncEngineCount</code> to the constant 8 on every device, so it counts no copy hardware. The
        stepper's engine switch is a CUDA idea; on AMD, measure.
      </p>
      <Beyond>
        <p>
          The book uses CUDA only. HIP and Metal facts here are checked against vendor sources in notes/today/ch23-facts.md (B3,
          B4, C1, C4, C5) and infra-facts.md (B8).
        </p>
      </Beyond>
      <Ask q={questionsS233['ch23/23.3/q-hip-engines']} />
      <p>
        On Apple silicon there is nothing to copy: a shared buffer is the GPU's memory. What is left of the pipeline is CPU
        work, writing the next chunk's input into a buffer and reading a finished chunk's output. Apple's documented way to
        overlap that with the GPU is several instances of a buffer in flight: the CPU fills one while the GPU reads another, a
        semaphore stops the CPU from running ahead, and a completion handler releases an instance when its command buffer is
        done. The rule that keeps it correct: "finalize all CPU writes to one buffer instance before you commit a command buffer
        that references it".
      </p>
      <p>
        Switch the stepper to its Metal view. The lanes become "CPU: drain, then fill" and "GPU", and S becomes K, the buffers
        in flight. With K = 1 nothing overlaps; with 2 the CPU's fills and drains can run under the GPU's kernels. Our Metal
        file waits for a buffer's last command buffer with <code>waitUntilCompleted</code> instead of Apple's semaphore and
        completion handler; either way the CPU blocks only when the buffer it needs is still in use.
      </p>
      <p>
        Notice what the Metal view has lost: a lane. One CPU thread does every fill and every drain, so the most overlap can
        do is hide the shorter of two lanes, fill + drain on the CPU and the kernel on the GPU. The read-out gives the bound,
        n · max(fill + drain, kernel) per chunk. Unless the kernel is long, the CPU&rsquo;s lane is the longer one: it moves
        the data twice. With the two equal, the best case, overlap at most halves the serial time; lab 23.1 measures how close
        a Mac gets.
      </p>
      <Today checked={`${CHECKED} (ch23-facts C1, C4)`}>
        <p>
          Two command queues "can" be scheduled concurrently: Apple's word, not a promise. No copy engine is documented. A queue
          from <code>newCommandQueue</code> allows at most 64 uncompleted command buffers;{' '}
          <code>newCommandQueueWithMaxCommandBufferCount:</code> changes that. K stays far below it here.
        </p>
      </Today>
      <Ask q={questionsS233['ch23/23.3/q-metal']} />

      <h2>Where you will meet this again</h2>
      <p>
        Pipeline parallelism in inference servers is this pipeline with groups of layers as the stages: while one stage works
        on a batch, the activations of the previous batch move on to the next stage. llama.cpp's default multi-GPU mode,{' '}
        <code>--split-mode layer</code>, is described as "pipelined": its scheduler keeps up to 4 copies of each input, records
        an event after each split and makes the next one wait on it. PyTorch's pipelining documentation calls the idle start
        the "pipeline startup bubble", the fill you watched in the stepper. vLLM uses pipeline parallelism across nodes.
      </p>
      <Beyond>
        <p>
          The mapping is ours (ch23-facts D1, D5, D7). In llama.cpp, the CUDA backend implements that record and wait as{' '}
          <code>cudaEventRecord</code> and <code>cudaStreamWaitEvent</code>: the tools of page{' '}
          <a href={href('ch23', '23.4')}>23.3–23.5</a>.
        </p>
      </Beyond>
    </>
  );
}
