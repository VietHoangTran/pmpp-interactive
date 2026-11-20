import { useEffect, useRef, useState, type ReactNode, type RefObject } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { fmt } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  DECODE_STEP,
  MD_STEP,
  SPEEDUPS,
  ceiling,
  moduleSpeedup,
  nearestIndex,
  overlapped,
  rebalance,
  speedup,
  type AppModule,
  type ModuleResult,
  type Place,
} from '../../sim/amdahl';
import {
  FILLS,
  HIP_DEFAULT_QUEUES,
  KERNEL_COUNTS,
  LAUNCH_RATIOS,
  compare,
  simulate,
  type LanesInput,
  type LanesResult,
  type Mode,
} from '../../sim/tasklanes';

// ---------------------------------------------------------------- formatting
/** A speed-up: "16.8×", "1,000×", or "unbounded". */
const times = (v: number) => (isFinite(v) ? fmt(Number(v.toPrecision(6))) + '×' : 'unbounded');
/** A fraction as a percentage, three significant digits: "5.95%". */
const pct = (v: number) => fmt(v * 100, 3) + '%';
/** A time in units of one kernel's duration d. */
const dur = (v: number) => fmt(v, 3) + ' d';

// The cases the page works with. Every number shown for them comes from the models.
const P_OVERLAP = { p: 0.9, s: 50 };
const BOOK = { p: 0.95, s: 100 };

// ---------------------------------------------------------------- small helpers (this section only)
/** Width of an element in CSS pixels, so a drawing keeps legible text on a phone. */
function useWidth(initial: number, min = 280): [RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(initial);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const read = () => setW(Math.max(min, Math.round(el.clientWidth)));
    read();
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, [min]);
  return [ref, w];
}

const SWATCH: Record<Place, string> = { host: 'var(--ink)', device: 'var(--ink-3)' };

// ---------------------------------------------------------------- explorer 1: which modules go on the GPU
type PresetKey = 'md' | 'decode';
const PRESETS: Record<PresetKey, { label: string; modules: AppModule[]; share: string }> = {
  md: { label: "The book's time step (pp. 537–538)", modules: MD_STEP, share: 'Share of the original time' },
  decode: { label: 'A decode step (ours)', modules: DECODE_STEP, share: "Share of today's step" },
};

/** One bar of spans placed in time; `scale` is the time the full width stands for. */
function SpanBar({ label, figure, spans, scale, lane }: {
  label: string;
  figure: ReactNode;
  spans: { module: number; start: number; end: number; place: Place }[];
  scale: number;
  lane: string;
}) {
  return (
    <div className="row" style={{ gap: '0.3rem 1rem', marginBottom: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
      <div style={{ flex: '0 0 11rem', lineHeight: 1.25 }}>
        <b>{label}</b>
        <div className="muted" style={{ fontSize: '0.76rem' }}>{figure}</div>
      </div>
      <div
        data-lane={lane}
        style={{
          flex: '1 1 14rem', position: 'relative', height: '1.6rem', background: 'var(--cell)',
          border: '1px solid var(--rule)', borderRadius: 2, overflow: 'hidden', minWidth: '12rem',
        }}
      >
        {spans
          .filter((s) => s.end > s.start)
          .map((s) => {
            const left = (s.start / scale) * 100;
            const width = ((s.end - s.start) / scale) * 100;
            return (
              <div
                key={s.module}
                data-span={s.module + 1}
                title={`module ${s.module + 1}`}
                style={{
                  position: 'absolute', top: 0, bottom: 0, left: `${left}%`, width: `${width}%`,
                  background: SWATCH[s.place], borderRight: '1px solid var(--page)', color: 'var(--page)',
                  fontSize: '0.72rem', fontWeight: 600, display: 'grid', placeItems: 'center', overflow: 'hidden',
                }}
              >
                {width >= 3.2 ? s.module + 1 : ''}
              </div>
            );
          })}
      </div>
    </div>
  );
}

function ModuleCard({
  m, i, shareLabel, onChange,
}: {
  m: AppModule;
  i: number;
  shareLabel: string;
  onChange: (patch: Partial<AppModule> & { share?: number }) => void;
}) {
  const onDevice = m.place === 'device';
  return (
    <div className="panel" data-module={i + 1} style={{ borderLeft: `5px solid ${SWATCH[m.place]}`, display: 'grid', gap: '0.55rem' }}>
      <b style={{ lineHeight: 1.25 }}>
        {i + 1}. {m.name}
      </b>
      <label className="field">
        <span>
          {shareLabel}: <b data-out="share">{pct(m.share)}</b>
        </span>
        <input
          type="range" min="0" max="100" step="0.5" value={Math.round(m.share * 200) / 2}
          aria-label={`share of ${m.name}`}
          onChange={(e) => onChange({ share: +e.target.value / 100 })}
        />
      </label>
      <div className="row" style={{ gap: '0.4rem 0.8rem' }}>
        <span className="seg" data-place>
          <button aria-pressed={!onDevice} disabled={m.fixed === 'device'} onClick={() => onChange({ place: 'host' })}>host</button>
          <button aria-pressed={onDevice} disabled={m.fixed === 'host'} onClick={() => onChange({ place: 'device' })}>GPU</button>
        </span>
        {m.fixed ? <span className="muted" style={{ fontSize: '0.78rem' }}>stays on the {m.fixed === 'host' ? 'host' : 'GPU'}</span> : null}
      </div>
      <label className="field" style={{ opacity: onDevice ? 1 : 0.55 }}>
        <span>
          On the GPU, <b data-out="s">{times(m.s)}</b> faster
        </span>
        <input
          type="range" min="0" max={SPEEDUPS.length - 1} step="1" value={nearestIndex(SPEEDUPS, m.s)} disabled={!onDevice}
          aria-label={`speed-up of ${m.name} on the GPU`}
          onChange={(e) => onChange({ s: SPEEDUPS[+e.target.value] })}
        />
      </label>
      <label className="check" style={{ opacity: onDevice ? 0.55 : 1 }}>
        <input type="checkbox" checked={!!m.chained} disabled={onDevice} aria-label={`${m.name} chained to the device`}
          onChange={(e) => onChange({ chained: e.target.checked })} />
        chained to the device
      </label>
    </div>
  );
}

function ModuleTimes({ mods, r }: { mods: AppModule[]; r: ModuleResult }) {
  let t = 0;
  const before = mods.map((m, i) => {
    const s = { module: i, start: t, end: t + m.share, place: m.place };
    t += m.share;
    return s;
  });
  const after = (lane: Place) => r.spans.filter((s) => s.lane === lane).map((s) => ({ ...s, place: lane }));
  return (
    <div className="panel" data-timebars style={{ marginBottom: '1rem' }}>
      <SpanBar label="Before" figure="the original run, 100%; colour shows where each module goes now" spans={before} scale={1} lane="before" />
      <SpanBar
        label="After: host"
        figure={<>stretched: the full width is <span data-out="after-total">{pct(r.total)}</span> of the original time</>}
        spans={after('host')} scale={r.total} lane="host"
      />
      <SpanBar label="After: GPU" figure="same scale as the host row" spans={after('device')} scale={r.total} lane="device" />
      <span className="legend" style={{ marginTop: '0.3rem' }}>
        <span><i style={{ background: SWATCH.host }} />runs on the host</span>
        <span><i style={{ background: SWATCH.device }} />runs on the GPU</span>
        <span>numbers are the modules below</span>
      </span>
    </div>
  );
}

function ModulesBench() {
  const [preset, setPreset] = useState<PresetKey>('md');
  const [mods, setMods] = useState<AppModule[]>(MD_STEP);
  const [overlap, setOverlap] = useState(false);
  const r = moduleSpeedup(mods, overlap);
  const load = (k: PresetKey) => {
    setPreset(k);
    setMods(PRESETS[k].modules);
    setOverlap(false);
  };
  const change = (i: number) => (patch: Partial<AppModule> & { share?: number }) =>
    setMods((ms) => {
      const { share, ...rest } = patch;
      const next = share === undefined ? ms : rebalance(ms, i, share);
      return next.map((m, j) => (j === i ? { ...m, ...rest } : m));
    });
  const unchanged = JSON.stringify(mods) === JSON.stringify(PRESETS[preset].modules);
  return (
    <Bench
      title="Which modules go on the GPU"
      task="Find a placement whose ceiling is above 20×, and say what it had to move."
      foot={
        <>
          The top bar is the original run, everything on the host at the module's share. The two rows below are the new
          run, stretched so the full width is its length; with overlap on, the GPU row runs beside the host row. A chained
          module feeds the GPU or waits for its result within one step, so it cannot overlap. Moving a share slider scales
          the other modules so the shares still add up to 100%. In the book's preset, the split of the 5% and the 4× for
          the three small modules are ours; the decode preset is entirely ours.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
          Preset
          <span className="seg" data-presets>
            {(Object.keys(PRESETS) as PresetKey[]).map((k) => (
              <button key={k} aria-pressed={preset === k} onClick={() => load(k)}>
                {PRESETS[k].label}
              </button>
            ))}
          </span>
        </div>
        <label className="check">
          <input type="checkbox" aria-label="host and device overlap" checked={overlap} onChange={(e) => setOverlap(e.target.checked)} />
          host and GPU overlap
        </label>
      </div>
      <ModuleTimes mods={mods} r={r} />
      <div className="cols plot">
        <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(13.5rem, 1fr))', gap: '0.7rem' }}>
          {mods.map((m, i) => (
            <ModuleCard key={i} m={m} i={i} shareLabel={PRESETS[preset].share} onChange={change(i)} />
          ))}
        </div>
        <div className="panel" data-readout>
          <p className="muted" style={{ margin: 0 }}>Application speed-up</p>
          <p style={{ fontSize: '1.9rem', fontWeight: 600, margin: '0 0 0.5rem', lineHeight: 1.15 }} data-out="overall">
            {times(r.speedup)}
          </p>
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <tr>
                <td>Host time, of the original</td>
                <td className="num" data-out="host">{pct(r.host)}</td>
              </tr>
              <tr>
                <td>GPU time, of the original</td>
                <td className="num" data-out="device">{pct(r.device)}</td>
              </tr>
              {overlap ? (
                <tr>
                  <td>GPU time hidden behind unchained host work</td>
                  <td className="num" data-out="hidden">{pct(r.hidden)}</td>
                </tr>
              ) : null}
              <tr>
                <td>New run time, of the original</td>
                <td className="num" data-out="total">{pct(r.total)}</td>
              </tr>
              <tr>
                <td>Ceiling, 1 / host time</td>
                <td className="num" data-out="ceiling">{times(r.ceiling)}</td>
              </tr>
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.6rem 0 0' }}>
            {preset === 'md' && unchanged
              ? "The book's setting. The page prints 17× without overlap (exactly 16.8×) and 20× with it."
              : r.host > 0 && r.total <= r.host * 1.0001
                ? 'The GPU is fully hidden: only the host part is left, so this is the ceiling.'
                : r.host === 0
                  ? 'Nothing is left on the host, so there is no ceiling from host work.'
                  : `The host part is ${pct(r.host / r.total)} of the new run time.`}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer 2: many small kernels
interface LanesState {
  k: number;
  f: number;
  ratio: number;
  mode: Mode;
  queues: number;
  pairs: boolean;
  concurrent: boolean;
  grouped: boolean;
}
const MODES: { mode: Mode; label: string }[] = [
  { mode: 'stream', label: 'One stream' },
  { mode: 'streams', label: 'A stream each' },
  { mode: 'encoder', label: 'Metal encoder' },
  { mode: 'graph', label: 'Graph' },
];
const compareKey = (s: LanesState) =>
  s.mode === 'streams' ? (isFinite(s.queues) ? 'streams4' : 'streams') : s.mode === 'encoder' ? (s.concurrent ? 'encoder' : 'serial') : s.mode;

function LanesPlot({ r, scale }: { r: LanesResult; scale: number }) {
  const [ref, W] = useWidth(720);
  const labelW = W < 520 ? 66 : 96;
  const rowH = 22;
  const gap = 6;
  const top = 16;
  const occH = 34;
  const rowsN = 1 + r.rows.length;
  const occY = top + rowsN * (rowH + gap) + 4;
  const H = occY + occH + 34;
  const x0 = labelW;
  const x1 = W - 10;
  const X = (t: number) => x0 + (Math.min(t, scale) / scale) * (x1 - x0);
  const rowY = (j: number) => top + j * (rowH + gap);
  const step = scale <= 12 ? 1 : scale <= 30 ? 2 : scale <= 60 ? 5 : 10;
  const ticks: number[] = [];
  for (let t = 0; t <= scale + 1e-9; t += step) ticks.push(t);
  const occ = r.occupancy;
  const occPath =
    `M${X(0)},${occY + occH} ` +
    occ.map((o, j) => {
      const next = j + 1 < occ.length ? occ[j + 1].t : r.total;
      const y = occY + occH - o.fill * occH;
      return `L${X(o.t).toFixed(1)},${y.toFixed(1)} L${X(next).toFixed(1)},${y.toFixed(1)}`;
    }).join(' ') +
    ` L${X(r.total)},${occY + occH} Z`;
  return (
    <div ref={ref} style={{ width: '100%', overflow: 'hidden' }}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Timeline of the host and the device" style={{ display: 'block' }}>
        <rect x={x0} y={top - 4} width={x1 - x0} height={occY + occH - top + 4} fill="var(--page)" stroke="var(--rule)" />
        {ticks.map((t) => (
          <g key={t}>
            <line x1={X(t)} x2={X(t)} y1={top - 4} y2={occY + occH} stroke="var(--rule-soft)" />
            <text x={X(t)} y={occY + occH + 14} fontSize="11" textAnchor="middle">{t}</text>
          </g>
        ))}
        <text x={(x0 + x1) / 2} y={H - 4} fontSize="11.5" textAnchor="middle">time, in units of one kernel's duration d</text>
        {/* host lane */}
        <text x={x0 - 6} y={rowY(0) + rowH / 2 + 4} fontSize="11" textAnchor="end">host</text>
        {r.host.map((h, j) => {
          const w = X(h.end) - X(h.start);
          return (
            <g key={'h' + j} data-host={h.kind}>
              <rect x={X(h.start)} y={rowY(0)} width={Math.max(1, w - 0.6)} height={rowH} fill="var(--ink)" />
              {w >= 16 ? (
                <text x={X(h.start) + w / 2} y={rowY(0) + rowH / 2 + 4} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--page)' }}>
                  {h.kind === 'graph' ? (w >= 40 ? 'graph' : 'g') : h.i}
                </text>
              ) : null}
            </g>
          );
        })}
        {r.commit !== null ? (
          <g data-commit>
            <line x1={X(r.commit)} x2={X(r.commit)} y1={rowY(0) - 3} y2={rowY(0) + rowH + 3} stroke="var(--ink)" strokeWidth="2" />
            <text x={X(r.commit) + 4} y={rowY(0) - 2} fontSize="10.5">commit</text>
          </g>
        ) : null}
        {/* device rows */}
        {r.rows.map((label, j) => (
          <text key={'l' + j} x={x0 - 6} y={rowY(j + 1) + rowH / 2 + 4} fontSize="11" textAnchor="end">{label}</text>
        ))}
        {r.kernels.map((k) => {
          const w = X(k.end) - X(k.start);
          return (
            <g key={'k' + k.i} data-kernel={k.i}>
              <rect x={X(k.start)} y={rowY(k.row + 1)} width={Math.max(1, w - 1)} height={rowH} rx="2"
                fill="var(--ink-3)" stroke={k.dependsOn !== null ? 'var(--ink)' : 'none'} strokeDasharray={k.dependsOn !== null ? '3 2' : undefined} />
              {w >= 16 ? (
                <text x={X(k.start) + w / 2} y={rowY(k.row + 1) + rowH / 2 + 4} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--page)' }}>
                  {k.i}
                </text>
              ) : null}
            </g>
          );
        })}
        {r.barriers.map((b, j) => (
          <g key={'b' + j} data-barrier>
            <line x1={X(b)} x2={X(b)} y1={rowY(1) - 3} y2={rowY(rowsN) - 3} stroke="var(--ink)" strokeWidth="1.4" strokeDasharray="4 3" />
            {j === 0 ? <text x={X(b) + 3} y={rowY(1) - 4} fontSize="10">barrier</text> : null}
          </g>
        ))}
        {/* occupancy */}
        <text x={x0 - 6} y={occY + occH / 2 - 2} fontSize="11" textAnchor="end">device</text>
        <text x={x0 - 6} y={occY + occH / 2 + 11} fontSize="11" textAnchor="end">in use</text>
        <line x1={x0} x2={x1} y1={occY} y2={occY} stroke="var(--rule)" strokeDasharray="2 3" />
        <text x={x1 - 3} y={occY + 10} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>100%</text>
        <path d={occPath} fill="var(--ink-3)" fillOpacity="0.45" stroke="var(--ink-3)" strokeWidth="1" />
      </svg>
    </div>
  );
}

function ManyKernelsBench() {
  const [s, setS] = useState<LanesState>({ k: 8, f: 0.25, ratio: 0.2, mode: 'stream', queues: Infinity, pairs: false, concurrent: true, grouped: false });
  const set = (patch: Partial<LanesState>) => setS((v) => ({ ...v, ...patch }));
  const base = { k: s.k, f: s.f, d: 1, L: s.ratio, pairs: s.pairs, grouped: s.grouped };
  const input: LanesInput = { ...base, mode: s.mode, queues: s.queues, concurrent: s.concurrent };
  const r = simulate(input);
  const all = compare(base);
  const scale = Math.max(...all.map((x) => x.total));
  const current = compareKey(s);
  const segs = <T,>(name: string, value: T, options: { v: T; label: string }[], onPick: (v: T) => void, data: string) => (
    <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
      {name}
      <span className="seg" data-seg={data}>
        {options.map((o) => (
          <button key={o.label} aria-pressed={o.v === value} onClick={() => onPick(o.v)}>{o.label}</button>
        ))}
      </span>
    </div>
  );
  return (
    <Bench
      title="Many small kernels"
      task="Find a setting where streams beat one stream by 4× or more, then one where nothing but the graph helps."
      foot={
        <>
          Host row: one mark per launch, per encoded Metal dispatch, or the graph's single launch. Device rows: one per
          stream or hardware queue, or the encoder's dispatches as they overlap. A dashed outline marks a kernel that reads
          the one before it; dashed vertical lines are Metal barriers. The bottom strip is the fraction of the device in use.
          The time axis is the same for every mode, so switching modes compares like with like.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <span>Kernels k = <b data-out="k">{s.k}</b></span>
          <input type="range" aria-label="number of kernels" min="0" max={KERNEL_COUNTS.length - 1} step="1"
            value={KERNEL_COUNTS.indexOf(s.k)} onChange={(e) => set({ k: KERNEL_COUNTS[+e.target.value] })} style={{ width: '11rem' }} />
        </label>
        <label className="field">
          <span>Each fills f = <b data-out="f">{pct(s.f)}</b> of the device</span>
          <input type="range" aria-label="fill fraction" min="0" max={FILLS.length - 1} step="1"
            value={FILLS.indexOf(s.f)} onChange={(e) => set({ f: FILLS[+e.target.value] })} style={{ width: '11rem' }} />
        </label>
        <label className="field">
          <span>Launch cost L = <b data-out="ratio">{fmt(s.ratio)}</b> d</span>
          <input type="range" aria-label="launch cost over kernel duration" min="0" max={LAUNCH_RATIOS.length - 1} step="1"
            value={LAUNCH_RATIOS.indexOf(s.ratio)} onChange={(e) => set({ ratio: LAUNCH_RATIOS[+e.target.value] })} style={{ width: '11rem' }} />
        </label>
        {segs('Dependencies', s.pairs, [{ v: false, label: 'none' }, { v: true, label: 'pairs' }], (v) => set({ pairs: v }), 'deps')}
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        {segs('How they are issued', s.mode, MODES.map((m) => ({ v: m.mode, label: m.label })), (v) => set({ mode: v }), 'mode')}
        {s.mode === 'streams'
          ? segs('Hardware queues', s.queues, [{ v: Infinity, label: 'unlimited' }, { v: HIP_DEFAULT_QUEUES, label: `${HIP_DEFAULT_QUEUES} (HIP's default)` }], (v) => set({ queues: v }), 'queues')
          : null}
        {s.mode === 'encoder'
          ? segs('Dispatch type', s.concurrent, [{ v: true, label: 'concurrent' }, { v: false, label: 'serial' }], (v) => set({ concurrent: v }), 'dispatch')
          : null}
        {s.mode === 'encoder' && s.concurrent && s.pairs
          ? segs('Encode order', s.grouped, [{ v: false, label: 'as issued' }, { v: true, label: 'producers first' }], (v) => set({ grouped: v }), 'order')
          : null}
      </div>
      <div className="panel" style={{ marginBottom: '1rem' }}>
        <LanesPlot r={r} scale={scale} />
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 3fr)' }}>
        <div className="panel" data-readout>
          <p className="muted" style={{ margin: 0 }}>Time for all {s.k} kernels</p>
          <p style={{ fontSize: '1.7rem', fontWeight: 600, margin: '0 0 0.5rem', lineHeight: 1.15 }} data-out="total">{dur(r.total)}</p>
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <tr>
                <td>Device in use, on average</td>
                <td className="num" data-out="util">{pct(r.utilisation)}</td>
              </tr>
              <tr>
                <td>Most kernels running at once</td>
                <td className="num" data-out="max">{r.maxConcurrent}</td>
              </tr>
              <tr>
                <td>Device idle while the host is still issuing</td>
                <td className="num nowrap" data-out="idle">{dur(r.idleWhileLaunching)}</td>
              </tr>
              {r.barriers.length ? (
                <tr>
                  <td>Barriers</td>
                  <td className="num" data-out="barriers">{r.barriers.length}</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <table className="data" style={{ margin: 0 }} data-compare>
            <thead>
              <tr>
                <th>Same kernels, issued as</th>
                <th className="num">Time</th>
                <th className="num">In use</th>
                <th className="num">At once</th>
              </tr>
            </thead>
            <tbody>
              {all.map((x) => (
                <tr key={x.key} data-key={x.key} style={x.key === current ? { background: 'var(--rule-soft)', fontWeight: 600 } : undefined}>
                  <td>{x.label}</td>
                  <td className="num nowrap" data-cell="total">{dur(x.total)}</td>
                  <td className="num">{pct(x.utilisation)}</td>
                  <td className="num">{x.maxConcurrent}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS223: Record<string, Question> = {
  'ch22/22.3/q-when-concurrent': {
    kind: 'choice',
    id: 'ch22/22.3/q-when-concurrent',
    prompt: (
      <>
        You have six small, independent kernels in one step and put each on its own stream so they can run at the same
        time. When does that make the step shorter?
      </>
    ),
    options: [
      {
        label: 'When each kernel, running alone, leaves much of the device idle.',
        correct: true,
      },
      {
        label: 'Always: kernels on different streams run side by side.',
        why: 'Different streams make overlap possible, not certain. A kernel that fills the device leaves nothing for the others.',
      },
      {
        label: 'Only when the kernels share inputs, so the caches stay warm.',
        why: 'Shared data is not the condition. Independent kernels that each use a small part of the device are.',
      },
    ],
    explain: (
      <p>
        Overlap fills parts of the device that one kernel leaves idle. If each kernel already occupies the whole device,
        the next one waits for it on any stream: set f to 100% in the explorer and no mode finishes in less than k·d. The book's condition is that each small kernel alone uses the GPU poorly (p. 538). AMD states
        the same one: concurrency "is most effective when the kernels do not fully utilize the GPU's resources".
      </p>
    ),
    source: 'ch. 22 §22.3, p. 538; notes/today/ch22-facts.md B1',
  },
  'ch22/22.3/q-hip-queues': {
    kind: 'numeric',
    id: 'ch22/22.3/q-hip-queues',
    prompt: (
      <>
        On ROCm with default settings, a program creates 8 streams on one GPU and puts a small kernel on each. At most how
        many hardware queues carry those kernels?
      </>
    ),
    answer: 4,
    explain: (
      <p>
        <code>GPU_MAX_HW_QUEUES</code> defaults to 4: the HIP runtime creates at most that many hardware queues per device
        per process, and streams beyond them reuse the same queues round-robin. In the explorer, a stream each with 4
        queues stops at 4 kernels at once however small f is (the model assumes a shared queue runs its kernels in order,
        which AMD's page does not spell out). CUDA's guide gives a different limit, 128 resident grids per device.
      </p>
    ),
    source: 'notes/today/ch22-facts.md B1, A2 (HIP docs at ROCm 10.2; CUDA Programming Guide 13.5)',
  },
  'ch22/22.3/q-default-stream': {
    kind: 'choice',
    id: 'ch22/22.3/q-default-stream',
    prompt: <>Which of these keeps independent kernels from overlapping by default, even when the kernels themselves are fine?</>,
    options: [
      {
        label: "CUDA's legacy default stream, used when you launch with no stream, and Metal's plain compute encoder.",
        correct: true,
      },
      {
        label: 'Streams created with cudaStreamNonBlocking or hipStreamNonBlocking, one per kernel, on one device.',
        why: 'Those flags exist precisely so that the stream does not synchronise with the default stream.',
      },
      {
        label: 'A Metal compute encoder made with dispatch type concurrent, with no barrier between the dispatches.',
        why: "That is Metal's documented way to let dispatches overlap; you add barriers only where one reads another's output.",
      },
    ],
    explain: (
      <>
        <p>
          CUDA: a kernel on the legacy default stream begins only after all earlier work on the device has completed,
          and nothing in any stream starts until it has finished; non-blocking streams are exempt. HIP: AMD calls the
          default stream "a blocking operation" and offers <code>hipStreamNonBlocking</code>; whether it blocks exactly as
          CUDA's legacy stream does is unverified. Metal: the encoder the plain call returns dispatches its commands
          serially.
        </p>
      </>
    ),
    source: 'notes/today/ch22-facts.md A2, B1; ch02-facts.md C9',
  },
  'ch22/22.3/q-metal-overlap': {
    kind: 'choice',
    id: 'ch22/22.3/q-metal-overlap',
    prompt: (
      <>
        On Metal, dispatches A and B are independent and should overlap; dispatch C reads B's output. What does Apple
        document for this?
      </>
    ),
    options: [
      {
        label: 'One compute encoder of dispatch type concurrent, with memoryBarrier(scope:) encoded before C.',
        correct: true,
      },
      {
        label: 'A and B on two different command queues, which the GPU runs at the same time.',
        why: 'Apple documents ordering within a queue. It does not document that two queues run at the same time.',
      },
      {
        label: 'A plain compute encoder, which overlaps any dispatches that do not conflict.',
        why: "The plain encoder is serial by Apple's documentation: it runs A, B and C one after another.",
      },
    ],
    explain: (
      <p>
        The barrier makes every dispatch encoded before it finish, and its writes visible, before anything after it
        starts. It waits for A as well as B. That is why the order you encode in matters: try pairs with the Metal encoder
        in the explorer, as issued and producers first. With the Metal 4 types every resource is untracked, so placing the
        barriers is always your job.
      </p>
    ),
    source: 'notes/today/ch22-facts.md C1; infra-facts.md C12; ch02-facts.md C9',
  },
  'ch22/22.3/q-graphs': {
    kind: 'choice',
    id: 'ch22/22.3/q-graphs',
    prompt: (
      <>
        PyTorch documents <code>torch.compile</code>'s "reduce-overhead" mode, which uses CUDA graphs, as "useful for small
        batches". Why small batches?
      </>
    ),
    options: [
      {
        label: "Each kernel is short, so a launch's fixed CPU cost is a larger share of the step.",
        correct: true,
      },
      {
        label: 'A graph lets each kernel use less memory, which matters most at small batch.',
        why: 'A graph changes how work is submitted, not what a kernel stores. PyTorch names CPU overhead as the main gain.',
      },
      {
        label: 'Small batches cannot use streams, so a graph is their only way to overlap.',
        why: 'Streams work at any batch size. The graph removes per-launch host cost; it is not about overlap.',
      },
    ],
    explain: (
      <p>
        It is Amdahl's law with the launch as the remainder. NVIDIA: for a kernel with a short execution time, the launch
        overhead "can be a significant fraction" of the whole; a graph pays it once for the sequence. AMD calls the
        overhead of a direct HIP launch negligible and the layers a framework adds significant, and PyTorch's graphs skip
        exactly those layers. In the explorer, set L above d and compare one stream with the graph.
      </p>
    ),
    source: 'notes/today/ch22-facts.md D7, A3, B2',
  },
  'ch22/22.3/q-decode-kernels': {
    kind: 'numeric',
    id: 'ch22/22.3/q-decode-kernels',
    prompt: (
      <>
        Per token, kernels take 9 parts of the time and host work (sampling, launches) takes 1 part. The kernels become 3
        times faster and nothing overlaps. By what factor do tokens per second rise?
      </>
    ),
    answer: 2.5,
    tol: 0.01,
    unit: '×',
    explain: (
      <p>
        Before: 9 + 1 = 10 parts. After: 9/3 + 1 = 4 parts. 10 / 4 = {fmt(speedup(0.9, 3))}. The host part was a tenth of
        each token and is now a quarter of it, so a second 3× on the kernels buys only 4 / 2 = 2× more, and no kernel speed
        gets past {times(ceiling(0.9))} overall.
      </p>
    ),
    source: 'the method of ch. 22 §22.3, pp. 537–538; the host sampler is ch01-facts.md D2',
  },
  'ch22/22.3/q-host-share': {
    kind: 'numeric',
    id: 'ch22/22.3/q-host-share',
    prompt: (
      <>
        Host and device overlap, and the device's part is completely hidden behind the host's. What is the largest share
        of the original run time the host part can have if the application must reach at least 20×?
      </>
    ),
    answer: 5,
    tol: 0.001,
    unit: '%',
    explain: (
      <p>
        With the device hidden, the run time is the host part alone, so the speed-up is 1 / (host share). 1 / h ≥ 20 means
        h ≤ 5%. The book's example sits exactly on that line, which is why it reaches {times(overlapped(BOOK.p, BOOK.s))} and
        no more. Making the kernel faster cannot move it.
      </p>
    ),
    source: 'ch. 22 §22.3, p. 538',
  },
};

// ---------------------------------------------------------------- the section
export function S223() {
  const pPlain = speedup(P_OVERLAP.p, P_OVERLAP.s);
  const pOver = overlapped(P_OVERLAP.p, P_OVERLAP.s);
  return (
    <>
      <p>
        Every chapter so far has made one kernel faster. A program is rarely one kernel. It is several modules that hand
        data to each other, and they differ enormously in how much work they do. This section applies chapter 1's
        Amdahl's law to that picture: you decide module by module what goes on the GPU, and the modules you leave behind
        set the limit on what users see.
      </p>
      <Source section="22.3" pages="537" />

      <h2>An application is several modules</h2>
      <p>
        The book's example is a molecular-dynamics simulation, advanced one time step at a time. Each step builds a
        neighbour list, which feeds the non-bonded forces; computes several kinds of force on every atom; and then updates
        positions and velocities. Each kind of force has its own method: vibrational, rotational and non-bonded. The
        non-bonded forces involve interactions among many atoms and cost far more than the other two, so the modules tend
        to be written as separate passes over the force data, and each pass is a separate decision.
      </p>
      <Source section="22.3" pages="537" figs="Fig. 22.2" />
      <p>
        The outcome the book calls plausible: a kernel for the non-bonded forces, while the vibrational and rotational
        forces stay on the host. The update may stay on the host too, combining the host's forces with the device's and
        computing the new positions and velocities.
      </p>
      <Aside>
        <p>
          The page says the kernel computes the non-bonding forces "for all the grid points", and the host its forces "for
          the grid points". The example is about atoms; read "atoms". The same paragraph alternates "non-bonded" and
          "non-bonding" for one kind of force.
        </p>
      </Aside>

      <h2>The book's numbers</h2>
      <p>
        Say the non-bonded forces take 95% of the original sequential run time, and the kernel makes them 100 times
        faster. The other 5% stays on the host at its old speed. The new run time is 5% + 95%/100 = 5.95% of the old one,
        and the speed-up is 1 / 5.95% = {times(speedup(BOOK.p, BOOK.s))}. The page prints 17×.
      </p>
      <Source section="22.3" pages="537–538" />
      <p>
        Then the book lets host and device work at the same time. While the kernel computes non-bonded forces, the host
        computes the others. The kernel's 0.95% fits inside the host's 5% and stops counting, and the speed-up becomes
        1 / 5% = {times(overlapped(BOOK.p, BOOK.s))}. That is also the ceiling. While 5% of the work runs on the host at
        its old speed, no kernel gets the application past 20×, however fast it is. The kernel was made 100 times faster
        and then hidden completely, and the application kept a fifth of that.
      </p>
      <Predict
        id="ch22/22.3/p-overlap"
        prompt={
          <>
            A different program: 90% of its time is made 50 times faster on the GPU and the other 10% stays on the host.
            Host and device do not overlap. What is the application's speed-up?
          </>
        }
        unit="×"
      >
        {(v) => (
          <>
            <p>
              1 / (10% + 90%/50) = 1 / (10% + 1.8%) = <b>{times(pPlain)}</b>.{' '}
              {Math.abs(v - pPlain) < 0.1
                ? 'That is what you had.'
                : Math.abs(v - pOver) < 0.2
                  ? "10 is the overlapped figure. Without overlap the GPU's 1.8% adds to the host's 10%."
                  : v >= 40
                    ? "50 is the speed-up of the part, not of the program: the 10% left alone counts at full size."
                    : 'Add the two remaining parts first, then take the reciprocal.'}
            </p>
            <p>
              Let them overlap and the GPU's 1.8% hides inside the host's 10%: {times(pOver)}, which is also the ceiling
              1 / 10% = {times(ceiling(P_OVERLAP.p))}. A faster kernel would change neither number.
            </p>
          </>
        )}
      </Predict>
      <p>
        In the explorer, each module of the time step is a box with its share of the original time, where it runs, how
        much faster it is on the GPU, and whether it is chained to the device. A chained module feeds the device or waits
        for its result within the step, as the arrows of Fig. 22.2 say: the non-bonded kernel needs the neighbour list,
        and the update needs both kinds of force. A chained module cannot overlap with the device. The book gives only the
        95/5 split; how the 5% divides between the other three modules, and the 4× they would get on a GPU, are ours.
      </p>
      <Simplification>
        <p>
          As in chapter 1's model: shares are fixed, host modules keep their old speed, and moving a module costs no copies
          and no launches. With overlap on, the run time is the chained host time plus the longer of the other host time
          and the GPU time. The book assumes the GPU hides completely; here it does, because the kernel's 0.95% fits
          inside the 2% of vibrational and rotational forces.
        </p>
      </Simplification>
      <ModulesBench />
      <p>Three things to find, each one or two changes away from the book's setting:</p>
      <ul>
        <li>Push the non-bonded speed-up from 100× to 1,000×. How much does the application gain, and why so little?</li>
        <li>Turn overlap on, then move the vibrational and rotational forces to the GPU. Why does overlap stop adding anything?</li>
        <li>Find a placement whose ceiling is above 20×. What did it have to move?</li>
      </ul>
      <p>
        The pattern behind all three: once the big module is fast, the small ones decide. Overlap can at best hide the
        device's time behind the host's; it never shortens the host's own. Only moving host work to the device, or making
        it faster where it is, raises the ceiling.
      </p>

      <h2>Many small activities</h2>
      <p>
        The book draws the wider lesson from the 5%. In a real application that remainder is rarely one module. It is many
        small activities, none worth a kernel of its own, and their accumulated time becomes the limit on the speed-up
        users see.
      </p>
      <Source section="22.3" pages="537" />
      <p>
        Its answer is task-level parallelism: when the data set is large enough, run the small activities in parallel with
        each other. A multi-core host can run them on different cores. Or the GPU can run several small kernels at once,
        one per activity: each alone may use the device poorly and gain little over the CPU, but together they can fill it.
        CUDA expresses this with streams, which{' '}
        <a href={href('ch23')}>chapter 23</a> covers in §23.3 (pp. 553–563).
      </p>
      <Source section="22.3" pages="538" />
      <p>
        Whether it pays depends on two things the book leaves implicit: how much of the device each kernel leaves idle, and
        how fast the host can issue the kernels. The explorer has both. There are k independent kernels, each filling a
        fraction f of the device for a time d, and the host spends L on each launch. Times are in units of d. "Pairs" makes
        every second kernel read the output of the one before it.
      </p>
      <Simplification>
        <p>
          Kernels run side by side while their fills add up to at most 100%, otherwise they wait; CUDA's "may run
          concurrently" is modelled as "does". A launch, and the encoding of a Metal dispatch, cost the same fixed L. The
          graph is captured from one stream: one L, then its kernels back to back. A hardware queue shared by several
          streams runs its kernels in launch order (our assumption). The Metal command buffer starts at its commit, after
          the last dispatch. No memory contention, no copies, no cost for a barrier. Times are relative, not device times.
        </p>
      </Simplification>
      <ManyKernelsBench />
      <p>Try these before reading on:</p>
      <ul>
        <li>Set f to 100%. Which modes still gain anything over one stream?</li>
        <li>Set f to 10%, pick a stream each, then switch to 4 queues. Where does the overlap stop?</li>
        <li>Raise L above d. Which mode wins, and why do more streams not help?</li>
        <li>Choose pairs and the Metal encoder. Compare encoding as issued with producers first.</li>
      </ul>
      <Ask q={questionsS223['ch22/22.3/q-when-concurrent']} />
      <Ask q={questionsS223['ch22/22.3/q-hip-queues']} />

      <h2>Several kernels at once, on each backend</h2>
      <p>
        The book says only that CUDA supports this with streams. Each backend today has its own way to let independent
        kernels overlap, its own limit, and its own default that prevents it. None of them promises overlap: CUDA's guide
        says work on different streams may run concurrently when there are no dependencies between them and enough free
        resources.
      </p>
      <Today checked="2026-11-20 (ch22-facts A2, B1, C1; infra-facts C12; ch02-facts C9)">
        <p>
          Sources: the CUDA Programming Guide and Best Practices Guide 13.5, the HIP documentation at ROCm 10.2, and
          Apple's Metal documentation with the macOS 27.0 SDK headers.
        </p>
      </Today>
      <table className="data wide" data-backends>
        <thead>
          <tr>
            <th />
            <th>To let independent kernels overlap</th>
            <th>Limit and condition</th>
            <th>What prevents it by default</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><b>CUDA</b></td>
            <td>
              Launch them on non-default streams. <code>concurrentKernels</code> in <code>cudaDeviceProp</code> says whether
              the device can.
            </td>
            <td>
              Up to 128 resident grids per device, for every compute capability the guide lists (7.5 to 12.x). Overlap
              needs no event dependency, no implicit synchronisation and enough resources.
            </td>
            <td>
              The legacy default stream. A kernel launched there waits for all earlier work on the device, and nothing
              starts in any stream until it finishes. Streams made with <code>cudaStreamNonBlocking</code> are exempt.
            </td>
          </tr>
          <tr>
            <td><b>HIP</b></td>
            <td>The same: streams, and the same <code>concurrentKernels</code> field.</td>
            <td>
              At most <code>GPU_MAX_HW_QUEUES</code> hardware queues per device per process, 4 by default; further streams
              share them round-robin. AMD: concurrency "is most effective when the kernels do not fully utilize the GPU's
              resources".
            </td>
            <td>
              The default stream, which AMD calls "a blocking operation"; <code>hipStreamNonBlocking</code> makes one that is
              not. Whether it blocks exactly as CUDA's legacy stream does is unverified.
            </td>
          </tr>
          <tr>
            <td><b>Metal</b></td>
            <td>
              One compute encoder of dispatch type concurrent, with <code>memoryBarrier(scope:)</code> before any dispatch
              that reads another's output. A barrier waits for every dispatch encoded before it.
            </td>
            <td>
              Apple documents no overlap between command queues. With the Metal 4 (<code>MTL4</code>) types every resource
              is untracked, so you place every barrier yourself.
            </td>
            <td>The encoder the plain call returns: its dispatches run one after another.</td>
          </tr>
        </tbody>
      </table>
      <Ask q={questionsS223['ch22/22.3/q-default-stream']} />
      <Ask q={questionsS223['ch22/22.3/q-metal-overlap']} />

      <h2>The launch is a small activity too</h2>
      <p>
        One small activity the book does not name comes with every kernel: the launch. Setting up and issuing a kernel
        costs host time. Behind a long kernel it disappears; in front of a short one it does not. In the explorer, once L
        is larger than d the device spends most of its time waiting for the host, however many streams you give it.
      </p>
      <Beyond>
        <p>Sources read 2026-11-20: notes/today/ch22-facts.md A3, B2, C2 and D7.</p>
      </Beyond>
      <p>
        NVIDIA's guide says it plainly: for a kernel with a short execution time, the launch overhead "can be a significant
        fraction of the overall end-to-end execution time". Its remedy is a CUDA graph. You record a sequence of launches
        once, by capturing a stream or by building the graph with its API, instantiate it once, and then launch the whole
        sequence as one unit as often as you need, paying the setup per graph rather than per kernel. HIP graphs have the
        same shape with the <code>hip</code> prefix. AMD frames the cost differently: negligible for a kernel launched
        directly through HIP, significant when a framework adds layers of redirection on the way. Metal has indirect
        command buffers, which Apple describes as a way to encode commands once, reuse them, and reduce CPU overhead.
        Calling them Metal's counterpart of graphs is our analogy: an indirect command buffer is a list of commands with no
        dependency edges, and Apple gives no figure for what it saves.
      </p>
      <p>
        Real code relies on this. llama.cpp builds use CUDA graphs by default (the standalone ggml library does not) and
        HIP graphs as well. PyTorch's <code>torch.compile</code> has a mode, "reduce-overhead", that does it with CUDA
        graphs. Set L above d in the explorer and compare one stream with the graph.
      </p>
      <Ask q={questionsS223['ch22/22.3/q-graphs']} />

      <h2>A decode step is an application</h2>
      <p>
        Generating text is the same picture at a smaller scale. Each step runs the projections and attention, a string of
        small elementwise kernels with a launch for each, a sampler that turns logits into a token, and host code that
        turns the token back into text. In llama.cpp the default sampler chain runs on the host over an array of candidate
        tokens; an experimental path moves sampling into the backend's graph. A sampler on the host is an Amdahl remainder
        of a particular kind. It does not shrink when the kernels get faster, and the next step cannot start until it has
        finished, so overlap cannot hide it.
      </p>
      <Beyond>
        <p>
          Ours, not the book's. llama.cpp's sampler: notes/today/ch01-facts.md D2, read 2026-11-20. The decode preset's
          shares are invented for illustration, not measured.
        </p>
      </Beyond>
      <p>
        Pick "A decode step (ours)" in the module explorer above. Make both kinds of kernel 10 times faster and read the
        ceiling. Turn overlap on and see what it hides and what it cannot. Then move sampling to the GPU.
      </p>
      <p>
        That is why this course puts the end of the decode step on the device: argmax in milestone M4 (
        <a href={href('ch10', 'labs')}>lab 10.3</a>) and sampling in <a href={href('x1')}>module X1</a>. What has to reach
        the host each step is then a token id. If the next step also reads that id from device memory, the host never
        waits for it, and the work left on the host, launches and text, runs while the device works.
      </p>
      <Ask q={questionsS223['ch22/22.3/q-decode-kernels']} />

      <h2>Real backends decide overlap kernel by kernel</h2>
      <p>
        On Metal, llama.cpp and MLX both encode a model's kernels into concurrent compute encoders and add a buffer barrier
        only where one is needed. llama.cpp's rule, as its source states it: a node may run alongside the nodes encoded
        before it if it writes nothing they read or wrote and reads nothing they wrote; otherwise a barrier goes in and the
        tracking starts over. MLX sets a flag when an output buffer was an input of the previous group of dispatches, and
        inserts the barrier then. That is the question of <a href={href('ch22', '22.2')}>section 22.2</a>, who writes and
        who reads, asked of whole kernels instead of threads.
      </p>
      <Today checked="2026-11-20 on the default branches (ch22-facts D8)">
        <p>
          llama.cpp:{' '}
          <a href="https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-metal" target="_blank" rel="noreferrer">
            ggml/src/ggml-metal
          </a>
          ; concurrency is on unless <code>GGML_METAL_CONCURRENCY_DISABLE</code> is set. MLX:{' '}
          <a href="https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal" target="_blank" rel="noreferrer">
            mlx/backend/metal
          </a>
          . Both change often: read the directory, not a line number.
        </p>
      </Today>
      <p>
        When you read either backend, keep the explorer's pairs in mind. A barrier waits for everything encoded before it,
        so the order the nodes are encoded in decides how much work each barrier holds back.
      </p>

      <h2>Check</h2>
      <p>One more question on the section's arithmetic, the one to remember when someone quotes a kernel's speed-up.</p>
      <Ask q={questionsS223['ch22/22.3/q-host-share']} />
    </>
  );
}
