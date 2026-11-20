import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  ALGOS,
  ALGO_LABEL,
  N_MAX_LOG,
  N_MIN_LOG,
  brentKungUpTo,
  compare,
  curve,
  hybridLag,
  koggeStoneFrom,
  networkWinner,
  overallWinner,
  type AlgoId,
  type CurvePoint,
} from '../../sim/algoselect';
import {
  BINS,
  BOOK_SHARE,
  DISTRIBUTIONS,
  DIST_LABEL,
  GRID,
  POINTS_PER_BIN,
  bestCapacity,
  binWith,
  capacityForShare,
  countsFor,
  histogram,
  work,
  type Distribution,
} from '../../sim/binning';
import { CHARGE_LABEL, type Charge } from '../../sim/prefixnet';
import { widthChoices } from '../../sim/warps';

/** A caption above a group of buttons (a <label> would forward clicks to the first button). */
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const WRAP: CSSProperties = { flexWrap: 'wrap' };
const int = (v: number) => Math.round(v).toLocaleString('en-US');
const dec = (v: number, d = 2) => Number(v.toFixed(d)).toLocaleString('en-US');
const pct = (v: number) => (v === 0 ? '0' : v < 0.001 ? '<0.1' : dec(100 * v, 1));

/**
 * True while the viewport is at most 600 px wide. The plots then use a narrower drawing so
 * that their labels stay legible on a phone. (Chapter 10 has the same helper; a shared one
 * would belong in src/lib.)
 */
function useNarrow(px = 600): boolean {
  const query = `(max-width: ${px}px)`;
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && window.matchMedia(query).matches);
  useEffect(() => {
    const m = window.matchMedia(query);
    const on = () => setNarrow(m.matches);
    on();
    m.addEventListener('change', on);
    return () => m.removeEventListener('change', on);
  }, [query]);
  return narrow;
}

// ---------------------------------------------------------------- explorer 1: which scan
const CHARGES: Charge[] = ['adds', 'warps'];
const CHARGE_NOTE: Record<Charge, string> = {
  adds: 'Only the threads that add are charged: a step with a additions costs ⌈a / P⌉ time units.',
  warps:
    'A warp with at least one adding lane is charged all its lanes, the reason chapter 11 gives for real resource use above the formulas.',
  threads: '',
};
const DASH: Record<AlgoId, string | undefined> = { 'kogge-stone': undefined, 'brent-kung': '7 4', coarsened: '2 3' };
const STROKE: Record<AlgoId, string> = { 'kogge-stone': 'var(--ink)', 'brent-kung': 'var(--ink)', coarsened: 'var(--ink-2)' };

function Bar({ value, max }: { value: number; max: number }) {
  const w = max > 0 ? Math.max(1, (value / max) * 100) : 0;
  return (
    <span aria-hidden="true" style={{ display: 'block', height: 6, marginTop: 3, background: 'var(--rule-soft)', borderRadius: 2 }}>
      <span style={{ display: 'block', height: 6, width: `${w}%`, background: 'var(--ink-2)', borderRadius: 2 }} />
    </span>
  );
}

function Legend() {
  return (
    <div className="ui" style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem 1.1rem', fontSize: '0.85em', margin: '0 0 0.3rem' }}>
      {ALGOS.map((a) => (
        <span key={a} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
          <svg width="28" height="8" aria-hidden="true">
            <line x1="1" x2="27" y1="4" y2="4" stroke={STROKE[a]} strokeWidth={a === 'coarsened' ? 2.4 : 1.8} strokeDasharray={DASH[a]} />
          </svg>
          {ALGO_LABEL[a]}
        </span>
      ))}
    </div>
  );
}

function TimePlot({ pts, n, p, bkUpTo, ksFrom }: { pts: CurvePoint[]; n: number; p: number; bkUpTo: number | null; ksFrom: number | null }) {
  const narrow = useNarrow();
  const W = narrow ? 380 : 640;
  const H = narrow ? 260 : 290;
  const pad = { l: 50, r: 12, t: 10, b: 42 };
  const ys = pts.flatMap((q) => ALGOS.map((a) => q[a]));
  const yMin = Math.max(1, Math.min(...ys) / 1.4);
  const yMax = Math.max(...ys) * 1.4;
  const lx = (v: number) => Math.log2(v);
  const X = (v: number) => pad.l + (lx(v) / Math.max(1, lx(n))) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * (H - pad.t - pad.b);
  const every = narrow ? (pts.length > 9 ? 4 : 2) : pts.length > 9 ? 2 : 1;
  const xt = pts.filter((_, i) => i % every === 0).map((q) => q.p);
  const yt: number[] = [];
  for (let e = Math.ceil(Math.log10(yMin)); e <= Math.floor(Math.log10(yMax)); e++) yt.push(10 ** e);
  return (
    <div>
    <Legend />
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Time units against execution units" style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {bkUpTo ? (
        <rect x={pad.l} y={pad.t} width={Math.max(0, X(bkUpTo) - pad.l)} height={H - pad.t - pad.b} fill="var(--rule)" opacity="0.35" data-region="bk" />
      ) : null}
      {xt.map((v) => (
        <g key={'x' + v}>
          <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(v)} y={H - pad.b + 15} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
            {v >= 1024 ? v / 1024 + 'k' : v}
          </text>
        </g>
      ))}
      {yt.map((v) => (
        <g key={'y' + v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
            {v >= 1000 ? v / 1000 + 'k' : v}
          </text>
        </g>
      ))}
      <text x={(W + pad.l) / 2} y={H - 8} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
        execution units P
      </text>
      <text x={14} y={(H - pad.b + pad.t) / 2} fontSize="11.5" textAnchor="middle" transform={`rotate(-90 14 ${(H - pad.b + pad.t) / 2})`} style={{ fill: 'var(--ink-2)' }}>
        time units
      </text>
      {ksFrom ? (
        <g data-mark="ks-from">
          <line x1={X(ksFrom)} x2={X(ksFrom)} y1={pad.t} y2={H - pad.b} stroke="var(--ink-3)" strokeDasharray="3 3" />
        </g>
      ) : null}
      {ALGOS.map((a) => (
        <polyline
          key={a}
          data-curve={a}
          points={pts.map((q) => `${X(q.p)},${Y(q[a])}`).join(' ')}
          fill="none"
          stroke={STROKE[a]}
          strokeWidth={a === 'coarsened' ? 2.4 : 1.8}
          strokeDasharray={DASH[a]}
        />
      ))}
      <line x1={X(p)} x2={X(p)} y1={pad.t} y2={H - pad.b} stroke="var(--ink)" strokeWidth="1.2" />
      {ALGOS.map((a) => {
        const q = pts.find((x) => x.p === p);
        return q ? <circle key={'c' + a} cx={X(p)} cy={Y(q[a])} r="3.6" fill="var(--page)" stroke={STROKE[a]} strokeWidth="1.6" /> : null;
      })}
    </svg>
    </div>
  );
}

const PRESETS = [
  { key: 'few', label: '1,024 elements, 32 units', logN: 10, logP: 5 },
  { key: 'many', label: '1,024 elements, 1,024 units', logN: 10, logP: 10 },
  { key: 'big', label: '16,384 elements, 1,024 units', logN: 14, logP: 10 },
];

function ScanChoice() {
  const { reports } = useReports();
  const widths = useMemo(() => widthChoices(reports as Reports), [reports]);
  const [logN, setLogN] = useState(10);
  const [logP, setLogP] = useState(5);
  const [charge, setCharge] = useState<Charge>('adds');
  const [wKey, setWKey] = useState('book');
  const warp = (widths.find((c) => c.key === wKey) ?? widths[0]).width;
  const n = 2 ** logN;
  const lp = Math.min(logP, logN);
  const p = 2 ** lp;
  const rows = compare(n, p, charge, warp);
  const pts = useMemo(() => curve(n, charge, warp), [n, charge, warp]);
  const bkUpTo = useMemo(() => brentKungUpTo(n, charge, warp), [n, charge, warp]);
  const ksFrom = useMemo(() => koggeStoneFrom(n, charge, warp), [n, charge, warp]);
  const net = networkWinner(rows);
  const all = overallWinner(rows);
  const lag = hybridLag(rows);
  const maxTime = Math.max(...ALGOS.map((a) => rows[a].time));
  const maxAdds = Math.max(...ALGOS.map((a) => rows[a].additions));

  return (
    <Bench
      title="Which scan for this hardware?"
      task="Start at the question's case. Then give the block more units, one doubling at a time, until Kogge-Stone wins."
      foot={
        <>
          Every count comes from chapter 11's models: the networks from section 11.2 and 11.10, the coarsened scan from
          section 11.6 (2N/T + log₂T − 1 steps for T threads, one per unit). A step with more charged lanes than units
          takes several rounds. Model: <code>src/sim/algoselect.ts</code>.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <span>
            N = <b data-value="n">{int(n)}</b> elements
          </span>
          <input
            type="range"
            min={N_MIN_LOG}
            max={N_MAX_LOG}
            value={logN}
            aria-label="Elements, as a power of two"
            onChange={(e) => setLogN(Number(e.target.value))}
            style={{ width: '12rem' }}
          />
        </label>
        <label className="field">
          <span>
            P = <b data-value="p">{int(p)}</b> execution units
          </span>
          <input
            type="range"
            min={0}
            max={logN}
            value={lp}
            aria-label="Execution units, as a power of two"
            onChange={(e) => setLogP(Number(e.target.value))}
            style={{ width: '12rem' }}
          />
        </label>
        <span style={FIELD}>
          Charge each step for
          <span className="seg" role="group" aria-label="Charge each step for" style={WRAP}>
            {CHARGES.map((c) => (
              <button key={c} aria-pressed={charge === c} onClick={() => setCharge(c)}>
                {CHARGE_LABEL[c]}
              </button>
            ))}
          </span>
        </span>
        <label className="field">
          Warp width
          <select value={wKey} aria-label="warp width" disabled={charge !== 'warps'} onChange={(e) => setWKey(e.target.value)}>
            {widths.map((c) => (
              <option key={c.key} value={c.key}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span style={FIELD}>
          Presets
          <span className="seg" role="group" aria-label="Presets" style={WRAP}>
            {PRESETS.map((q) => (
              <button
                key={q.key}
                onClick={() => {
                  setLogN(q.logN);
                  setLogP(q.logP);
                }}
              >
                {q.label}
              </button>
            ))}
          </span>
        </span>
      </div>
      <p className="muted" style={{ margin: '0 0 0.7rem' }} data-readout="charge">
        {CHARGE_NOTE[charge]}
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0, minWidth: '30rem' }} data-part="scan-table">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>Scan</th>
              <th className="num">time units on P</th>
              <th className="num">additions</th>
              <th className="num">steps, unlimited units</th>
              <th className="num">threads</th>
            </tr>
          </thead>
          <tbody>
            {ALGOS.map((a) => (
              <tr key={a} data-row={a} data-time={rows[a].time} data-adds={rows[a].additions}>
                <td style={{ paddingLeft: '0.6rem' }}>{ALGO_LABEL[a]}</td>
                <td className="num">
                  {int(rows[a].time)}
                  <Bar value={rows[a].time} max={maxTime} />
                </td>
                <td className="num">
                  {int(rows[a].additions)}
                  <Bar value={rows[a].additions} max={maxAdds} />
                </td>
                <td className="num">{int(rows[a].span)}</td>
                <td className="num">{int(rows[a].threads)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="cols plot" style={{ marginTop: '0.9rem' }}>
        <TimePlot pts={pts} n={n} p={p} bkUpTo={bkUpTo} ksFrom={ksFrom} />
        <div className="panel" aria-live="polite" data-readout="winner" data-network={net} data-overall={all}>
          <p style={{ margin: 0 }}>
            Faster of the two networks:{' '}
            {net === 'tie' ? (
              <span className="chip" data-value="network-winner">
                a tie
              </span>
            ) : (
              <span className="chip ok" data-value="network-winner">
                {ALGO_LABEL[net]}
              </span>
            )}{' '}
            <span data-value="network-times">
              {net === 'kogge-stone'
                ? `${int(rows['kogge-stone'].time)} time units against Brent-Kung's ${int(rows['brent-kung'].time)}.`
                : net === 'brent-kung'
                  ? `${int(rows['brent-kung'].time)} time units against Kogge-Stone's ${int(rows['kogge-stone'].time)}.`
                  : `both take ${int(rows['brent-kung'].time)} time units.`}
            </span>
          </p>
          <p style={{ margin: '0.6rem 0 0' }} data-readout="hybrid">
            The coarsened scan: {int(rows.coarsened.time)} time units with {int(rows.coarsened.threads)}{' '}
            {rows.coarsened.threads === 1 ? 'thread, which makes it the sequential loop' : 'threads'};{' '}
            {lag > 0
              ? `${int(lag)} behind the better network.`
              : lag === 0
                ? 'level with the better network.'
                : `${int(-lag)} ahead of the better network.`}{' '}
            {all === 'coarsened' ? <span className="chip ok">fastest of the three</span> : null}
          </p>
          <p className="muted" style={{ margin: '0.6rem 0 0' }} data-readout="crossover" data-bk-upto={bkUpTo ?? ''} data-ks-from={ksFrom ?? ''}>
            Over every whole number of units, not only powers of two: Brent-Kung is faster for every P up to{' '}
            <b data-value="bk-upto">{bkUpTo == null ? 'none' : int(bkUpTo)}</b> (shaded), and Kogge-Stone for every P from{' '}
            <b data-value="ks-from">{ksFrom == null ? 'none' : int(ksFrom)}</b> (dashed line).
            {bkUpTo != null && ksFrom != null && ksFrom > bkUpTo + 1 ? ' In between they tie or take turns.' : ''}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer 2: bin capacity
const sOf = (v: number) => Math.round(10 ** (v / 10));
const S_DEFAULT_V = 13; // S = 20

function BinMap({ counts, capacity, fullest }: { counts: number[]; capacity: number; fullest: number }) {
  const cell = 12;
  const size = GRID * cell;
  return (
    <svg viewBox={`-1 -1 ${size + 2} ${size + 2}`} role="img" aria-label="Atoms per bin, over the domain" style={{ width: '100%', maxWidth: 230, display: 'block' }}>
      {counts.map((k, i) => {
        const x = (i % GRID) * cell;
        const y = Math.floor(i / GRID) * cell;
        const over = k > capacity;
        return (
          <rect
            key={i}
            x={x + 0.5}
            y={y + 0.5}
            width={cell - 1}
            height={cell - 1}
            fill="var(--global)"
            fillOpacity={fullest ? 0.06 + 0.94 * (k / fullest) : 0.06}
            stroke={over ? 'var(--ink)' : 'none'}
            strokeWidth={over ? 1.6 : 0}
            data-over={over ? '' : undefined}
          />
        );
      })}
    </svg>
  );
}

function Histogram({ counts, capacity, fullest }: { counts: number[]; capacity: number; fullest: number }) {
  const h = histogram(counts);
  const buckets = Math.min(48, fullest + 1);
  const width = Math.ceil((fullest + 1) / buckets);
  const nb = Math.ceil((fullest + 1) / width);
  const fit: number[] = new Array(nb).fill(0);
  const over: number[] = new Array(nb).fill(0);
  h.forEach((bins, k) => {
    const b = Math.floor(k / width);
    if (k > capacity) over[b] += bins;
    else fit[b] += bins;
  });
  const top = Math.max(1, ...fit.map((v, i) => v + over[i]));
  const W = 420;
  const H = 210;
  const pad = { l: 40, r: 10, t: 26, b: 40 };
  const bw = (W - pad.l - pad.r) / nb;
  const Y = (v: number) => H - pad.b - (v / top) * (H - pad.t - pad.b);
  const cx = pad.l + ((capacity + 0.5) / width) * bw;
  const ticks = [0, Math.round(fullest / 2), fullest];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Histogram of atoms per bin with the capacity" style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <line x1={pad.l} x2={W - pad.r} y1={H - pad.b} y2={H - pad.b} stroke="var(--rule)" />
      {fit.map((v, i) => (
        <g key={i}>
          {v ? <rect x={pad.l + i * bw + 0.5} y={Y(v + over[i])} width={Math.max(1, bw - 1)} height={Y(over[i]) - Y(v + over[i])} fill="var(--global)" /> : null}
          {over[i] ? <rect x={pad.l + i * bw + 0.5} y={Y(over[i])} width={Math.max(1, bw - 1)} height={H - pad.b - Y(over[i])} fill="var(--ink-2)" /> : null}
        </g>
      ))}
      {ticks.map((t) => (
        <text key={t} x={pad.l + ((t + 0.5) / width) * bw} y={H - pad.b + 14} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
          {t}
        </text>
      ))}
      <text x={(W + pad.l) / 2} y={H - 8} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
        atoms in a bin
      </text>
      <text x={pad.l - 6} y={Y(top) + 4} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
        {top}
      </text>
      <text x={pad.l - 6} y={H - pad.b} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
        0
      </text>
      <text x={12} y={(H - pad.b + pad.t) / 2} fontSize="11.5" textAnchor="middle" transform={`rotate(-90 12 ${(H - pad.b + pad.t) / 2})`} style={{ fill: 'var(--ink-2)' }}>
        bins
      </text>
      <line x1={cx} x2={cx} y1={pad.t} y2={H - pad.b} stroke="var(--ink)" strokeWidth="1.6" strokeDasharray="4 3" data-mark="capacity" />
      <line x1={cx} x2={cx} y1={pad.t - 8} y2={pad.t} stroke="var(--ink)" strokeWidth="1.6" />
      <text x={Math.max(pad.l, Math.min(cx - 30, W - pad.r - 80))} y={pad.t - 12} fontSize="11.5" style={{ fill: 'var(--ink)' }}>
        capacity {capacity}
      </text>
    </svg>
  );
}

function HBar({ label, value, max, fill, data }: { label: string; value: number; max: number; fill: string; data: string }) {
  const w = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div style={{ marginBottom: '0.45rem' }} data-bar={data} data-units={Math.round(value)}>
      <div className="ui" style={{ display: 'flex', justifyContent: 'space-between', gap: '0.6rem', fontSize: '0.9em' }}>
        <span>{label}</span>
        <span className="num">{int(value)}</span>
      </div>
      <span aria-hidden="true" style={{ display: 'block', height: 9, background: 'var(--rule-soft)', borderRadius: 2 }}>
        <span style={{ display: 'block', height: 9, width: `${Math.max(value > 0 ? 0.6 : 0, w)}%`, background: fill, borderRadius: 2 }} />
      </span>
    </div>
  );
}

function BinCapacity() {
  const [dist, setDist] = useState<Distribution>('pocket');
  const [atoms, setAtoms] = useState(4096);
  const [cap, setCap] = useState(Infinity);
  const [sv, setSv] = useState(S_DEFAULT_V);
  const S = sOf(sv);
  const counts = useMemo(() => countsFor(dist, atoms), [dist, atoms]);
  const fullest = Math.max(...counts);
  const c = Math.max(0, Math.min(cap, fullest));
  const b = binWith(counts, c);
  const w = work(counts, c, S);
  const wFull = work(counts, fullest, S);
  const c3 = capacityForShare(counts, BOOK_SHARE);
  const cBest = useMemo(() => bestCapacity(counts, S), [counts, S]);
  const scale = Math.max(wFull.serial, w.serial);
  const oneInS = BINS / S;

  return (
    <Bench
      title="Bin capacity and the overflow list"
      task="Start sized for the fullest bin. Drag the capacity down until the host's pass overtakes the device, then find the lowest total."
      foot={
        <>
          {GRID} × {GRID} bins, one block of {POINTS_PER_BIN} grid points (8 × 8) per bin, a 3 × 3 neighbourhood of bins.
          One unit is one test of one atom slot for one grid point. Atoms are generated from a fixed seed. Model:{' '}
          <code>src/sim/binning.ts</code>.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Atoms placed
          <span className="seg" role="group" aria-label="Distribution" style={WRAP}>
            {DISTRIBUTIONS.map((d) => (
              <button key={d} aria-pressed={dist === d} onClick={() => setDist(d)}>
                {DIST_LABEL[d]}
              </button>
            ))}
          </span>
        </span>
        <label className="field">
          <span>
            <b data-value="atoms">{int(atoms)}</b> atoms
          </span>
          <input type="range" min={1024} max={16384} step={1024} value={atoms} aria-label="Atoms" onChange={(e) => setAtoms(Number(e.target.value))} style={{ width: '11rem' }} />
        </label>
        <label className="field">
          <span>
            S = <b data-value="speedup">{S}</b>: device tests per host test
          </span>
          <input type="range" min={0} max={30} value={sv} aria-label="Device speed-up over the host" onChange={(e) => setSv(Number(e.target.value))} style={{ width: '11rem' }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <span>
            Capacity c = <b data-value="capacity">{c}</b> atoms per bin
          </span>
          <input type="range" min={0} max={fullest} value={c} aria-label="Bin capacity" onChange={(e) => setCap(Number(e.target.value))} style={{ width: '16rem' }} />
        </label>
        <span style={FIELD}>
          Set the capacity
          <span className="seg" role="group" aria-label="Set the capacity" style={WRAP}>
            <button onClick={() => setCap(fullest)} aria-pressed={c === fullest}>
              fullest bin ({fullest})
            </button>
            <button onClick={() => setCap(c3)} aria-pressed={c === c3}>
              smallest under 3% ({c3})
            </button>
            <button onClick={() => setCap(cBest)} aria-pressed={c === cBest}>
              lowest total time ({cBest})
            </button>
          </span>
        </span>
      </div>
      <div className="cols" style={{ alignItems: 'start' }}>
        <div className="panel">
          <p className="muted" style={{ margin: '0 0 0.4rem' }}>
            The domain: darker bins hold more atoms (bins live in global memory); outlined bins are over capacity.
          </p>
          <BinMap counts={counts} capacity={c} fullest={fullest} />
        </div>
        <div className="panel">
          <p className="muted" style={{ margin: '0 0 0.4rem' }}>
            Bins by how many atoms they hold. <span className="chip global">fits in its bin</span>{' '}
            <span className="chip" style={{ background: 'var(--ink-2)', color: 'var(--page)' }}>
              over capacity
            </span>
          </p>
          <Histogram counts={counts} capacity={c} fullest={fullest} />
        </div>
      </div>
      <div className="cols" style={{ marginTop: '0.9rem', alignItems: 'start' }}>
        <div className="panel" data-readout="memory">
          <p style={{ margin: 0 }}>
            <b>Memory for the bins</b>: {BINS} × {c} = <b data-value="slots">{int(b.slots)}</b> slots, of which{' '}
            <span data-value="real">{int(b.real)}</span> real atoms and <span data-value="dummies">{int(b.dummies)}</span>{' '}
            zero-charge dummies ({b.slots ? pct(b.dummies / b.slots) : '0'}%).
          </p>
          <p className="muted" style={{ margin: '0.4rem 0 0' }} data-readout="full-sizing">
            {c === fullest
              ? `This is the sizing for the fullest bin: every bin has room for ${fullest} atoms.`
              : `Sized for the fullest bin it would be ${BINS} × ${fullest} = ${int(BINS * fullest)} slots${b.slots ? `, ${dec((BINS * fullest) / b.slots, 1)} times as many` : ''}.`}
          </p>
          <p style={{ margin: '0.7rem 0 0' }} data-readout="overflow" data-share={b.overflowShare}>
            <b>Overflow list</b>: <b data-value="overflow">{int(b.overflow)}</b> atoms from{' '}
            <span data-value="overflowing-bins">{b.overflowingBins}</span> bins, <b data-value="share">{pct(b.overflowShare)}%</b> of all atoms{' '}
            {b.overflowShare <= BOOK_SHARE ? (
              <span className="chip ok">within the book's 3%</span>
            ) : (
              <span className="chip">above the book's 3%</span>
            )}
          </p>
          <p className="muted" style={{ margin: '0.4rem 0 0' }}>
            Raising c pays while more than about one bin in S overflows: about {dec(oneInS, 0)} of {BINS} here.
          </p>
        </div>
        <div className="panel" data-readout="work">
          <HBar label={`Device: ${int(w.device)} slot tests ÷ S`} value={w.deviceTime} max={scale} fill="var(--ink-2)" data="device" />
          <HBar label={`Host: ${int(b.overflow)} overflow atoms, one test at a time`} value={w.hostTime} max={scale} fill="var(--ink)" data="host" />
          <HBar label="Kernel, then the host pass" value={w.serial} max={scale} fill="var(--ink-3)" data="serial" />
          <p className="muted" style={{ margin: '0.5rem 0 0' }}>
            In host-test times. {w.device ? `${pct(w.wasted / w.device)}% of the device's tests are on dummies. ` : ''}
            Sized for the fullest bin: {int(wFull.serial)}. Hiding the host pass behind the next kernel (chapter 21's
            refinement) would leave the larger of the two: {int(w.overlapped)}.
          </p>
          <p style={{ margin: '0.5rem 0 0' }} data-readout="bottleneck">
            {w.hostTime > w.deviceTime ? (
              <span className="chip fail">the host's serial pass takes longer than the kernel</span>
            ) : (
              <span className="chip ok">the kernel takes longer than the host's pass</span>
            )}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const Q = compare(1024, 32, 'adds', 32);
const Q_BK = Q['brent-kung'].time;
const Q_KS = Q['kogge-stone'].time;
const KS_FROM_1024 = koggeStoneFrom(1024, 'adds', 32)!;
const BK_UPTO_1024 = brentKungUpTo(1024, 'adds', 32)!;

export const questionsS221: Record<string, Question> = {
  'ch22/22.1/q-scan-units': {
    kind: 'choice',
    id: 'ch22/22.1/q-scan-units',
    prompt: (
      <>
        A block scans 1,024 elements and can keep only 32 adders busy in a step. Counting additions only, which scan
        finishes first?
      </>
    ),
    options: [
      {
        label: `Brent-Kung, about ${Q_BK} time units against ${Q_KS}: with so few units, the total number of additions decides.`,
        correct: true,
      },
      {
        label: 'Kogge-Stone, because it takes 10 steps instead of 19, and the number of steps is what a scan waits for.',
        why: 'That holds when every addition of a step can run at once. With 32 units each of Kogge-Stone’s steps takes up to 32 rounds.',
      },
      {
        label: 'They tie, because both end up doing the same additions, only grouped into steps differently.',
        why: 'Kogge-Stone does 9,217 additions on 1,024 elements, Brent-Kung 2,036.',
      },
    ],
    explain: (
      <p>
        Kogge-Stone&rsquo;s 10 steps have between 512 and 1,023 additions each, so on 32 units each takes 16 to 32
        rounds: {Q_KS} in all. Brent-Kung&rsquo;s 19 steps shrink fast, most of them to a single round: {Q_BK}. Counting
        every whole number of units, Brent-Kung stays ahead up to {BK_UPTO_1024} and Kogge-Stone wins only from{' '}
        {KS_FROM_1024}, three quarters of a unit per element. The explorer below lets you check both.
      </p>
    ),
    source: 'ch. 22 §22.1, p. 530; ch. 11 §11.5 and §11.10, pp. 265–267, 282–285',
  },
  'ch22/22.1/q-sort-logits': {
    kind: 'choice',
    id: 'ch22/22.1/q-sort-logits',
    prompt: (
      <>
        You need 128,000 float logits sorted for top-p sampling. Radix sort is your fastest sort, but the book says it
        works only for some key types. Can you use it?
      </>
    ),
    options: [
      {
        label: 'Yes, after mapping each float’s bits to an unsigned key in the same order; what radix cannot take is an arbitrary comparison.',
        correct: true,
      },
      {
        label: 'No: radix sort needs integer keys and logits are floats, so this is exactly the case merge sort is for.',
        why: 'A bit transform turns each float into an unsigned integer with the same order, and then radix sort applies.',
      },
      {
        label: 'Yes, as they are: the bits of a float, read as an unsigned integer, already sort in the order of the floats.',
        why: 'Only for non-negative values. Negative floats have the sign bit set, so they would land above every positive one, in reverse order.',
      },
    ],
    explain: (
      <p>
        Set the sign bit of a non-negative float and invert every bit of a negative one: read as unsigned integers, the
        results are in the floats&rsquo; order. That is the transform CUB documents for its radix sort, and the one you
        met in <a href={href('ch14', '14.4')}>chapter 14</a>. Generality is what merge sort keeps: it sorts by any
        comparison you can write, including one for which nobody has an order-preserving bit encoding.
      </p>
    ),
    source: 'ch. 22 §22.1, p. 530; ch. 14 §14.4 lesson (ours) and notes/today/ch14-facts.md A3',
  },
  'ch22/22.1/q-overflow': {
    kind: 'choice',
    id: 'ch22/22.1/q-overflow',
    prompt: (
      <>
        Atoms crowd into a few bins. Compared with sizing every bin for the fullest one, what does a capacity near the
        typical count plus an overflow list cost?
      </>
    ),
    options: [
      {
        label: 'A short serial pass over the few overflow atoms after the kernel, for much less memory and far fewer empty slots scanned.',
        correct: true,
      },
      {
        label: 'Nothing at all: atoms that do not fit are dropped, just as the cutoff already drops the far atoms.',
        why: 'Far atoms are left out by design. Overflow atoms are near, and their contributions must be added to complete the result (p. 531).',
      },
      {
        label: 'More work on the device, because every block must now search the overflow list for atoms near its points.',
        why: 'The device never reads the list. It is processed after the kernel, by the host in chapter 21 (p. 527).',
      },
    ],
    explain: (
      <p>
        Every slot of every bin is scanned by every grid point in its neighbourhood, dummy or not, so the capacity
        multiplies the device&rsquo;s work and the memory. The overflow list moves the rare excess to a serial pass. It
        stays cheap only while it stays small: in the explorer, a capacity too low makes the host the bottleneck.
      </p>
    ),
    source: 'ch. 22 §22.1, p. 531; ch. 21 §21.5, p. 527',
  },
  'ch22/22.1/q-cutoff-axis': {
    kind: 'choice',
    id: 'ch22/22.1/q-cutoff-axis',
    prompt: (
      <>
        Of the four axes (steps, exposed parallelism, generality, accuracy), which two does cutoff summation trade
        against each other, compared with direct summation?
      </>
    ),
    options: [
      { label: 'Accuracy, given up a little, for steps, cut a great deal; both expose ample parallelism.', correct: true },
      {
        label: 'Parallelism for steps: the cutoff version has fewer grid points it can compute at once.',
        why: 'Both run one thread per grid point. The book says both expose ample parallelism (p. 530).',
      },
      {
        label: 'Generality for accuracy: the cutoff version works only for certain kinds of atoms.',
        why: 'It works for any atoms. What it needs is a physical law under which far contributions are small (p. 531).',
      },
    ],
    explain: (
      <p>
        Direct summation&rsquo;s work grows with the square of the volume; cutoff summation&rsquo;s grows linearly,
        because each grid point only sums the atoms near it. The price is the far atoms&rsquo; individual contributions,
        which the physics says are small and can be handled together. Section <a href={href('ch22', '22.4')}>22.4</a>{' '}
        calls this kind of move a reformulation.
      </p>
    ),
    source: 'ch. 22 §22.1, pp. 530–531',
  },
  'ch22/22.1/q-two-kernels': {
    kind: 'choice',
    id: 'ch22/22.1/q-two-kernels',
    prompt: (
      <>
        llama.cpp uses matrix-vector kernels for up to about 8 tokens and a matrix kernel above that, with per-GPU
        exceptions marked &ldquo;tuned on RTX 4090&rdquo;. Why a measured threshold rather than a derived one?
      </>
    ),
    options: [
      {
        label: 'Which algorithm wins depends on the hardware and the weight format, so the crossover has to be measured on each.',
        correct: true,
      },
      {
        label: 'The matrix-vector kernel is more accurate, so it is kept for the small batches where users notice errors.',
        why: 'Both compute the same products. Nothing in the selection is about accuracy.',
      },
      {
        label: 'It is a hardware limit: a GPU cannot run a matrix kernel on a batch of fewer than 8 rows.',
        why: 'A matrix kernel runs on one row too. It just does padded, wasted work, as M3 does at batch 3.',
      },
    ],
    explain: (
      <p>
        The book&rsquo;s rule is to choose the best compromise for the given hardware. Here the two algorithms differ in
        how they reuse weights and in how much decoding work they repeat per token, and where those costs cross depends
        on the device and on the format. A threshold tuned per GPU is that rule written into a dispatch function.
      </p>
    ),
    source: 'ch. 22 §22.1, p. 530; notes/today/ch22-facts.md D6',
  },
};

export function S221() {
  const q = questionsS221;
  return (
    <>
      <p>
        An algorithm, in the book&rsquo;s definition, is a procedure in which every step is stated precisely
        (definiteness), every step can be carried out by a computer (effective computability), and the whole is
        guaranteed to end (finiteness). What matters here is that most problems have several of them.
      </p>
      <Source section="22.1" pages="530" />
      <p>
        Algorithms for one problem differ along at least four axes: how many steps they take (their complexity), how much
        parallelism they expose, how general they are, and how accurate or numerically stable they are. Usually none wins
        on all four, so you choose the best compromise for the hardware you have. The book shows three of these
        trade-offs with chapters you have done, each a different pair of axes.
      </p>

      <h2>Fewer steps, or fewer additions</h2>
      <p>
        Chapter 11&rsquo;s two scan networks are the first pair. Kogge-Stone finishes in log₂N steps but does about
        N·log₂N additions. Brent-Kung does fewer than 2N additions but needs about twice as many steps. Steps decide the
        time only when every addition of a step can run at once. When the hardware has fewer adders than a step has
        additions, the step takes several rounds, and the total number of additions starts to count.
      </p>
      <Source section="22.1" pages="530" figs="ch. 11 §11.5 and §11.10" />
      <p>
        The book adds a way to soften the extra work: a hybrid, either two parallel algorithms combined or a parallel one
        combined with a cheaper sequential one through thread coarsening. Chapter 11&rsquo;s three-phase scan is the
        second kind. Each thread scans its own run of elements sequentially, one parallel scan joins the run totals, and
        each thread adds its predecessor&rsquo;s total.
      </p>
      <Ask q={q['ch22/22.1/q-scan-units']} label="Reason it out, then check it below" />
      <ScanChoice />
      <p>
        Three things to find. First, how many units Kogge-Stone needs before it wins: with 1,024 elements, counting
        additions, it is not ahead until P reaches {KS_FROM_1024}. Second, what whole-warp charging does with a small N,
        where idle lanes of a partly used warp cost as much as busy ones. Third, the coarsened scan: at every P it is
        level with the better network, ahead of it, or one time unit behind. It runs one thread per unit and gives each
        thread a sequential run, so it adapts its work to the hardware instead of committing to one network.
      </p>
      <Simplification>
        <p>
          Chapter 11&rsquo;s model: one unit per addition, a step costs ⌈charge / P⌉ time units, steps never overlap. No
          memory traffic and no cost for barriers. P is a power of two, because the coarsened scan as counted needs a
          power-of-two thread count.
        </p>
      </Simplification>

      <h2>General, or fast</h2>
      <p>
        Sorting gives the second pair. Odd-even sort does the most work of chapter 14&rsquo;s sorts but is the most
        trivial to parallelise. Radix sort can beat merge sort on complexity because it never compares keys, and it
        parallelises very well, but it works only for certain key types. Merge sort compares, so it takes any key type
        with a well-defined comparison.
      </p>
      <Source section="22.1" pages="530" figs="ch. 14" />
      <p>The keys you are most likely to sort are logits, and logits are floats.</p>
      <Ask q={q['ch22/22.1/q-sort-logits']} />

      <h2>Exact, or fast</h2>
      <p>
        The third pair comes from chapter 21&rsquo;s electrostatic map. Direct Coulomb summation adds every atom&rsquo;s
        contribution at every grid point. It is exact, but its work grows with the square of the volume, because the
        atoms and the grid points both grow with it, and for large systems that is too much even for a massively
        parallel device. Cutoff summation adds only the atoms within a fixed distance of each grid point. It gives up a
        little accuracy for a large gain in speed, and both expose ample parallelism.
      </p>
      <Source section="22.1" pages="530–531" />
      <p>
        Why the accuracy may be given up is a fact about physics, not about computing. Under the laws these grid
        calculations rest on, particles far from a grid point contribute very little, and their contributions can be
        handled together by a separate method of much lower complexity.
      </p>

      <h3>Why the parallel version is harder</h3>
      <p>
        The trade exists in sequential code too; parallel code makes it harder to collect. The simple sequential cutoff
        loop takes one atom at a time and visits the grid points within the cutoff of it, which is easy because the grid
        points sit in an array indexed by coordinate. Run that in parallel, one thread per atom, and the threads scatter:
        several of them add into the same grid point.
      </p>
      <Source section="22.1" pages="531" />
      <p>
        So the parallel version turns it around: one thread per grid point, gathering. To tell a thread which atoms are
        near without scanning all of them, the atoms are first sorted into bins, boxes of grid space. A grid point&rsquo;s
        neighbourhood is the set of bins that can hold atoms within its cutoff. All the threads of a block scan the atoms
        of their combined neighbourhood together, and each tests every atom against its own cutoff. Section{' '}
        <a href={href('ch22', '22.2')}>22.2</a> names the two arrangements: atom-centric is input-centric, grid-centric is
        output-centric.
      </p>

      <h3>Uneven bins</h3>
      <p>
        Bins do not fill evenly: some are empty, some crowded. Room in every bin for the most atoms any bin could hold is
        prohibitively expensive. The well-known fix is a capacity well below the worst case and an overflow list. During
        binning, an atom whose bin is already full goes on the list instead, and after the kernel those atoms are
        processed to add their missing contributions.
      </p>
      <Source section="22.1" pages="531" />
      <p>
        This page of the book does not say where the overflow list is processed. Chapter 21 does: the host runs a
        sequential cutoff pass over it. Chapter 21 also pads every bin to the capacity with dummy atoms of charge zero, so
        that all bins have one size and loads stay coalesced; the dummies cost memory, and they lengthen the scans of
        blocks whose bins are not full. With a small overflow share, under 3% for example, the host pass typically takes
        less time than the device.
      </p>
      <Source section="21.5" pages="527" />
      <BinCapacity />
      <p>
        Start with one dense pocket, sized for the fullest bin: most of the memory and most of the device&rsquo;s tests
        go to dummies. Drag the capacity toward the bulk of the histogram. Memory and device time fall, the overflow list
        grows slowly, and the total falls with them. Keep going and the host&rsquo;s serial pass overtakes the kernel.
        That is Amdahl&rsquo;s law arriving one section early: the serial part, however small it starts, ends up setting
        the time.
      </p>
      <Simplification>
        <p>
          Two dimensions, a fixed 3 × 3 neighbourhood, one block per bin. Every scanned slot costs one unit whether or not
          the atom is inside the cutoff, so there is no divergence. S is a free parameter, not a device figure. No memory
          traffic, no binning pass, no copies between host and device.
        </p>
      </Simplification>
      <p>
        Where is the best capacity? One more slot per bin adds a slot to every bin, so every block scans more; it takes
        atoms off the host only in the bins that still overflow. So it pays while more than about one bin in S still
        overflows, and the &ldquo;lowest total time&rdquo; button lands where that stops being true.
      </p>
      <Beyond>
        <p>This rule is ours, derived from the explorer&rsquo;s work model; the book gives only the 3% example.</p>
      </Beyond>
      <Ask q={q['ch22/22.1/q-overflow']} />

      <h2>Selection by measurement</h2>
      <p>
        The same reasoning picks kernels in shipping code. llama.cpp keeps two algorithms for one operation, a weight
        matrix times the tokens of a batch. Up to 8 tokens it runs matrix-vector kernels; above that, a matrix-matrix
        kernel. The threshold is not derived. It is lowered for some quantised weight types on particular GPUs, with
        comments naming the GPU it was tuned on, and it has separate branches for AMD. One reason is in the source itself:
        for some formats the matrix-vector kernel redoes the decoding of the weights for every token, so the matrix
        kernel wins sooner. Choosing by a measured crossover is the book&rsquo;s best compromise for the given hardware,
        written into a dispatch function.
      </p>
      <Today checked="2026-11-20 against llama.cpp's master branch (ch22-facts D6)">
        <p>
          CUDA and HIP share{' '}
          <a href="https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda" target="_blank" rel="noreferrer">
            <code>ggml-cuda/</code>
          </a>
          : <code>mmvf</code> and <code>mmvq</code> up to 8 tokens, with exceptions marked &ldquo;tuned on RTX
          4090&rdquo;. Metal,{' '}
          <a href="https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-metal" target="_blank" rel="noreferrer">
            <code>ggml-metal/</code>
          </a>
          : <code>mul_mv_ext</code> for 2 to 8 tokens, then <code>mul_mm</code>. Line numbers move.
        </p>
      </Today>
      <p>
        Lab 22.1 on the <a href={href('ch22', 'labs')}>labs page</a> cuts a decode-time matrix-vector product two ways;
        its gather version is a kernel of the first kind.
      </p>
      <Ask q={q['ch22/22.1/q-cutoff-axis']} />
      <Ask q={q['ch22/22.1/q-two-kernels']} />
    </>
  );
}
