import { useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { fmt } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BOOK_CASES,
  FRACTIONS,
  RATES,
  SPEEDUPS,
  THREADS,
  ceiling,
  corners,
  fractionIndex,
  fractionNeeded,
  kernelNeeded,
  levers,
  limits,
  limitsCurve,
  nearestIndex,
  remaining,
  speedup,
  speedupCurve,
  untouchedShare,
} from '../../sim/amdahl';
import { useWidth } from './useWidth';

// ---------------------------------------------------------------- formatting
/** A speed-up: "1.42×", "1,000×", or "unbounded". Rounded first, so that 999.9999 reads as 1,000. */
const times = (v: number) => (isFinite(v) ? fmt(Number(v.toPrecision(6))) + '×' : 'unbounded');
/** A share of something, as a percentage with four significant digits. */
const share = (v: number) => fmt(v * 100, 4) + '%';
/** A parallel fraction exactly as the slider holds it: "99%", "99.95%". */
const frac = (p: number) => Number((p * 100).toFixed(4)) + '%';
const count = (v: number) => v.toLocaleString('en-US', { maximumFractionDigits: 3 });

// The cases the page works with. Every number shown for them comes from the model.
const DEFAULT_P = 0.99;
const DEFAULTS = { threads: 1000, rate: 1 / 4, cap: 10 };
const book30 = BOOK_CASES[0];
const book99 = BOOK_CASES[2];
/** q-fivehundred: the part made this many times faster, and the overall figure asked for. */
export const FIVE_HUNDRED = { s: 1000, target: 500 };

// ---------------------------------------------------------------- plot helpers
const tick = (v: number) => (v >= 1000 ? v / 1000 + 'k' : String(v));
const path = (pts: [number, number][]) => pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');

function FractionSlider({ p, setP }: { p: number; setP: (p: number) => void }) {
  return (
    <label className="field">
      <span>
        Parallel fraction p = <b data-out="p">{frac(p)}</b>
      </span>
      <input
        type="range"
        aria-label="parallel fraction"
        min="0"
        max={FRACTIONS.length - 1}
        step="1"
        value={fractionIndex(p)}
        onChange={(e) => setP(FRACTIONS[+e.target.value])}
        style={{ width: '15rem' }}
      />
    </label>
  );
}

// ---------------------------------------------------------------- explorer 1: Amdahl's law
function TimeBars({ p, s }: { p: number; s: number }) {
  const r = remaining(p, s);
  const us = untouchedShare(p, s);
  const seg = (part: 'untouched' | 'accelerated', width: number) => (
    <div
      data-part={part}
      style={{ width: `${width * 100}%`, background: part === 'untouched' ? 'var(--ink)' : 'var(--ink-3)', height: '100%' }}
    />
  );
  const row = (id: string, name: string, figure: ReactNode, parts: ReactNode) => (
    <div className="row" style={{ gap: '0.3rem 1rem', marginBottom: '0.55rem', alignItems: 'center' }}>
      <div style={{ flex: '0 0 13.5rem', lineHeight: 1.25 }}>
        <b>{name}</b>
        <div className="muted" style={{ fontSize: '0.76rem' }}>{figure}</div>
      </div>
      <div
        data-bar={id}
        style={{
          flex: '1 1 14rem', display: 'flex', height: '1.5rem', background: 'var(--cell)',
          border: '1px solid var(--rule)', borderRadius: 2, overflow: 'hidden',
        }}
      >
        {parts}
      </div>
    </div>
  );
  return (
    <div className="panel" style={{ marginBottom: '1rem' }} data-timebars>
      {row('before', 'Before', 'the whole run: 100%', <>{seg('untouched', r.untouched)}{seg('accelerated', p)}</>)}
      {row(
        'after',
        'After, same scale',
        <><span data-out="left">{share(r.total)}</span> of the original time</>,
        <>{seg('untouched', r.untouched)}{seg('accelerated', r.accelerated)}</>,
      )}
      {row(
        'zoom',
        'After, magnified',
        <>the untouched part is <span data-out="share">{share(us)}</span> of it</>,
        <>{seg('untouched', us)}{seg('accelerated', 1 - us)}</>,
      )}
      <span className="legend" style={{ marginTop: '0.2rem' }}>
        <span><i style={{ background: 'var(--ink)' }} />left alone: 1 − p</span>
        <span><i style={{ background: 'var(--ink-3)' }} />the accelerated part: p before, p / s after</span>
      </span>
    </div>
  );
}

function AmdahlPlot({ p, s }: { p: number; s: number }) {
  const [ref, W] = useWidth(600);
  const H = Math.round(Math.min(330, Math.max(250, W * 0.56)));
  const pad = { l: 52, r: 10, t: 24, b: 44 };
  const slot = 80; // room at the right for "unbounded"
  const x1 = W - pad.r - slot; // where s = 1,000 sits
  const xInf = W - pad.r - 22;
  const X = (v: number) => (isFinite(v) ? pad.l + (Math.log10(v) / 3) * (x1 - pad.l) : xInf);
  const Y = (v: number) => H - pad.b - (Math.log10(Math.min(Math.max(v, 1), 10000)) / 4) * (H - pad.t - pad.b);
  const top = ceiling(p);
  const now = speedup(p, s);
  const curve = speedupCurve(p).map((q): [number, number] => [X(q.s), Y(q.overall)]);
  const px = X(s);
  const py = Y(now);
  return (
    <div ref={ref} style={{ width: '100%', overflow: 'hidden' }}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Overall speed-up against the speed-up of the accelerated part" style={{ display: 'block' }}>
        <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
        {/* past s = 1,000 the axis is not to scale: one column for the unbounded case */}
        <rect x={x1 + 14} y={pad.t} width={W - pad.r - x1 - 14} height={H - pad.t - pad.b} fill="var(--rule-soft)" />
        {[1, 10, 100, 1000].map((v) => (
          <g key={'x' + v}>
            <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
            <text x={X(v)} y={H - pad.b + 15} fontSize="11" textAnchor="middle">{v.toLocaleString('en-US')}</text>
          </g>
        ))}
        <text x={W - pad.r} y={H - pad.b + 15} fontSize="11" textAnchor="end">unbounded</text>
        {[1, 10, 100, 1000, 10000].map((v) => (
          <g key={'y' + v}>
            <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
            <text x={pad.l - 6} y={Y(v) + 4} fontSize="11" textAnchor="end">{tick(v)}</text>
          </g>
        ))}
        <text x={(pad.l + x1) / 2} y={H - 8} fontSize="11.5" textAnchor="middle">speed-up of the accelerated part, s</text>
        <text x={14} y={(H - pad.b + pad.t) / 2} fontSize="11.5" textAnchor="middle" transform={`rotate(-90 14 ${(H - pad.b + pad.t) / 2})`}>overall speed-up</text>
        {/* what the whole run would gain if nothing were left alone */}
        <line x1={X(1)} y1={Y(1)} x2={X(1000)} y2={Y(1000)} stroke="var(--ink-3)" strokeWidth="1" strokeDasharray="2 4" />
        {/* the label gives way when the ceiling line would run through it */}
        {Math.abs(Y(top) - (Y(1000) - 11)) < 13 ? null : (
          <text x={X(1000) - 6} y={Y(1000) - 7} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>if p were 100%</text>
        )}
        {/* the ceiling 1 / (1 - p) */}
        <line x1={pad.l} x2={W - pad.r} y1={Y(top)} y2={Y(top)} stroke="var(--ink)" strokeWidth="1.2" strokeDasharray="6 4" />
        <text x={pad.l + 8} y={Y(top) - 6} fontSize="11.5" data-out="plot-ceiling">ceiling {times(top)}</text>
        {/* overall speed-up against s, and where it ends up when s has no bound */}
        <path d={path(curve)} fill="none" stroke="var(--ink)" strokeWidth="2.2" />
        <circle cx={xInf} cy={Y(top)} r="4.5" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.8" data-limit />
        {/* where the controls are now */}
        <line x1={px} x2={px} y1={py} y2={H - pad.b} stroke="var(--ink-3)" strokeWidth="1" />
        <circle cx={px} cy={py} r="5.5" fill="var(--ink)" stroke="var(--page)" strokeWidth="1.5" data-point />
      </svg>
    </div>
  );
}

function AmdahlBench({ p, setP }: { p: number; setP: (p: number) => void }) {
  const [finite, setFinite] = useState(100);
  const [unbounded, setUnbounded] = useState(false);
  const s = unbounded ? Infinity : finite;
  const now = speedup(p, s);
  const top = ceiling(p);
  const r = remaining(p, s);
  const need = kernelNeeded(100, p);
  const preset = (c: (typeof BOOK_CASES)[number]) => {
    setP(c.p);
    setUnbounded(!isFinite(c.s));
    if (isFinite(c.s)) setFinite(c.s);
  };
  return (
    <Bench
      title="Amdahl's law"
      task="Find the smallest parallel fraction that lets the overall figure reach 100× at all."
      foot="Log scales on both axes of the plot. Heavy line: overall speed-up; the filled dot is where the controls are now. Dashed line: the ceiling, 1 / (1 − p), which the open dot in the last column reaches. Dotted diagonal: what the run would gain if none of it were left alone. The bars are drawn to one scale except the last, which stretches the remaining time to full width."
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
          The book's cases (p. 9)
          <span className="seg" data-presets>
            {BOOK_CASES.map((c) => (
              <button key={c.label} aria-pressed={p === c.p && s === c.s} onClick={() => preset(c)}>
                {c.label}
              </button>
            ))}
          </span>
        </div>
        <FractionSlider p={p} setP={setP} />
        <label className="field">
          <span>
            Speed-up of that part s = <b data-out="s">{times(s)}</b>
          </span>
          <input
            type="range"
            aria-label="speed-up of the accelerated part"
            min="0"
            max={SPEEDUPS.length - 1}
            step="1"
            disabled={unbounded}
            value={nearestIndex(SPEEDUPS, finite)}
            onChange={(e) => setFinite(SPEEDUPS[+e.target.value])}
            style={{ width: '13rem' }}
          />
        </label>
        <label className="check">
          <input type="checkbox" aria-label="unbounded speed-up" checked={unbounded} onChange={(e) => setUnbounded(e.target.checked)} />
          unbounded
        </label>
      </div>
      <TimeBars p={p} s={s} />
      <div className="cols plot">
        <AmdahlPlot p={p} s={s} />
        <div className="panel" data-readout>
          <p className="muted" style={{ margin: 0 }}>Overall speed-up</p>
          <p style={{ fontSize: '1.9rem', fontWeight: 600, margin: '0 0 0.5rem', lineHeight: 1.15 }} data-out="overall">{times(now)}</p>
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <tr>
                <td>Ceiling, 1 / (1 − p)</td>
                <td className="num" data-out="ceiling">{times(top)}</td>
              </tr>
              <tr>
                <td>Run time left, of the original</td>
                <td className="num">{share(r.total)}</td>
              </tr>
              <tr>
                <td>Untouched part, as a share of the time left</td>
                <td className="num">{share(untouchedShare(p, s))}</td>
              </tr>
              <tr>
                <td>The s that 100× overall would need</td>
                <td className="num nowrap" data-out="needed">
                  {need === null ? 'none is enough' : isFinite(need) ? times(need) : 'unbounded'}
                </td>
              </tr>
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.6rem 0 0' }}>
            The third figure is also how much of the ceiling you have reached: the more of the remaining time is
            untouched, the less a faster s can add.
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer 2: what caps the kernel
function CapsPlot({ p, threads, rate, cap }: { p: number; threads: number; rate: number; cap: number }) {
  const [ref, W] = useWidth(600);
  const H = Math.round(Math.min(340, Math.max(260, W * 0.58)));
  const pad = { l: 52, r: 12, t: 24, b: 44 };
  const xLo = 1;
  const xHi = 100000;
  const yLo = 0.1;
  const yHi = 10000;
  const X = (v: number) => pad.l + ((Math.log10(Math.min(Math.max(v, xLo), xHi)) - Math.log10(xLo)) / 5) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(Math.min(Math.max(v, yLo), yHi)) - Math.log10(yLo)) / 5) * (H - pad.t - pad.b);
  const c = corners(rate, cap);
  const pts = limitsCurve(p, rate, cap);
  const now = limits(p, threads, rate, cap);
  const top = ceiling(p);
  const atSat = speedup(p, cap);
  const satRight = X(c.saturation) > W - pad.r - 120;
  return (
    <div ref={ref} style={{ width: '100%', overflow: 'hidden' }}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Overall speed-up against the number of threads" style={{ display: 'block' }}>
        <defs>
          <pattern id="s13-hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="7" stroke="var(--rule)" strokeWidth="2" />
          </pattern>
        </defs>
        <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
        {/* three stretches of the thread axis */}
        {c.breakEven > xLo ? <rect x={X(xLo)} y={pad.t} width={X(c.breakEven) - X(xLo)} height={H - pad.t - pad.b} fill="url(#s13-hatch)" data-region="slower" /> : null}
        <rect x={X(c.saturation)} y={pad.t} width={W - pad.r - X(c.saturation)} height={H - pad.t - pad.b} fill="var(--global-wash)" data-region="memory" />
        {[1, 10, 100, 1000, 10000, 100000].map((v) => (
          <g key={'x' + v}>
            <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
            <text x={X(v)} y={H - pad.b + 15} fontSize="11" textAnchor={v === xHi ? 'end' : 'middle'}>{tick(v)}</text>
          </g>
        ))}
        {[0.1, 1, 10, 100, 1000, 10000].map((v) => (
          <g key={'y' + v}>
            <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke={v === 1 ? 'var(--ink-3)' : 'var(--rule-soft)'} />
            <text x={pad.l - 6} y={Y(v) + 4} fontSize="11" textAnchor="end">{tick(v)}</text>
          </g>
        ))}
        <text x={(W + pad.l) / 2} y={H - 8} fontSize="11.5" textAnchor="middle">threads, n</text>
        <text x={14} y={(H - pad.b + pad.t) / 2} fontSize="11.5" textAnchor="middle" transform={`rotate(-90 14 ${(H - pad.b + pad.t) / 2})`}>speed-up over one CPU core</text>
        {/* the ceiling from the untouched part */}
        <line x1={pad.l} x2={W - pad.r} y1={Y(top)} y2={Y(top)} stroke="var(--ink)" strokeWidth="1.2" strokeDasharray="6 4" />
        <text x={pad.l + 8} y={Y(top) - 6} fontSize="11.5">ceiling {times(top)}</text>
        {/* the kernel alone: min(n r, M) */}
        <path d={path(pts.map((q): [number, number] => [X(q.threads), Y(q.s)]))} fill="none" stroke="var(--ink-2)" strokeWidth="1.2" />
        {/* the label gives way when the ceiling line would run through it */}
        {Math.abs(Y(top) - (Y(cap) - 10)) < 12 ? null : (
          <text x={W - pad.r - 6} y={Y(cap) - 6} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>kernel alone {times(cap)}</text>
        )}
        {/* the whole run */}
        <path d={path(pts.map((q): [number, number] => [X(q.threads), Y(q.overall)]))} fill="none" stroke="var(--ink)" strokeWidth="2.4" />
        {/* the two corners */}
        {c.breakEven > xLo ? (
          <g data-corner="break-even">
            <circle cx={X(c.breakEven)} cy={Y(1)} r="4.5" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.8" />
            {cap >= 4 ? <text x={X(c.breakEven) + 8} y={Y(1) + 16} fontSize="11">break-even</text> : null}
          </g>
        ) : null}
        <g data-corner="saturation">
          <circle cx={X(c.saturation)} cy={Y(atSat)} r="4.5" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.8" />
          <text
            x={satRight ? W - pad.r - 6 : X(c.saturation) + 8}
            y={H - pad.b - 8}
            fontSize="11"
            textAnchor={satRight ? 'end' : 'start'}
            style={{ fill: 'var(--global)' }}
          >
            memory saturated
          </text>
        </g>
        {/* where the controls are now */}
        <line x1={X(threads)} x2={X(threads)} y1={Y(now.overall)} y2={H - pad.b} stroke="var(--ink-3)" strokeWidth="1" />
        <circle cx={X(threads)} cy={Y(now.overall)} r="5.5" fill="var(--ink)" stroke="var(--page)" strokeWidth="1.5" data-point />
      </svg>
    </div>
  );
}

function CapsBench({ p, setP }: { p: number; setP: (p: number) => void }) {
  const [threads, setThreads] = useState(DEFAULTS.threads);
  const [rate, setRate] = useState(DEFAULTS.rate);
  const [cap, setCap] = useState(DEFAULTS.cap);
  const now = limits(p, threads, rate, cap);
  const c = corners(rate, cap);
  const lv = levers(p, threads, rate, cap);
  const top = ceiling(p);
  const reset = () => {
    setP(DEFAULT_P);
    setThreads(DEFAULTS.threads);
    setRate(DEFAULTS.rate);
    setCap(DEFAULTS.cap);
  };
  const tickAt = SPEEDUPS.indexOf(10) / (SPEEDUPS.length - 1);
  const lever = (id: string, label: string, v: number) => (
    <tr data-lever={id}>
      <td>{label}</td>
      <td className="num nowrap">{times(v)}</td>
      <td className="muted nowrap">{v <= lv.now * 1.001 ? 'no change' : `${fmt(v / lv.now)} times more`}</td>
    </tr>
  );
  const goal = (target: number) => (
    <span className={now.overall > target ? 'chip ok' : 'chip'} data-goal={target} style={now.overall > target ? undefined : { border: '1px solid var(--rule)' }}>
      past {target}×: {now.overall > target ? 'reached' : 'not yet'}
    </span>
  );
  return (
    <Bench
      title="What caps the kernel"
      task="From the defaults, get the overall figure past 50×, then past 100×. Note which control you had to move each time."
      foot="Log scales on both axes. Heavy line: the whole run. Thin line: the kernel alone, min(n × r, M). Dashed line: the ceiling set by the untouched part. Hatched: too few threads to match the CPU core. Shaded: thread counts at which memory is saturated."
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-start' }}>
        <label className="field">
          <span>
            Threads n = <b data-out="n">{count(threads)}</b>
          </span>
          <input
            type="range"
            aria-label="threads"
            min="0"
            max={THREADS.length - 1}
            step="1"
            value={nearestIndex(THREADS, threads)}
            onChange={(e) => setThreads(THREADS[+e.target.value])}
            style={{ width: '15rem' }}
          />
        </label>
        <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
          One GPU thread against the CPU core, r
          <span className="seg" data-rates>
            {RATES.map((o) => (
              <button key={o.label} aria-pressed={rate === o.value} onClick={() => setRate(o.value)}>
                {o.label}
              </button>
            ))}
          </span>
        </div>
        <label className="field" style={{ position: 'relative', paddingBottom: '1.15rem' }}>
          <span>
            Memory cap M = <b data-out="M">{times(cap)}</b>
          </span>
          <input
            type="range"
            aria-label="memory cap"
            min="0"
            max={SPEEDUPS.length - 1}
            step="1"
            value={nearestIndex(SPEEDUPS, cap)}
            onChange={(e) => setCap(SPEEDUPS[+e.target.value])}
            style={{ width: '17rem' }}
          />
          <span
            data-tick="book"
            style={{
              position: 'absolute', bottom: 0, left: `calc(${tickAt * 100}% + ${(8 - 16 * tickAt).toFixed(1)}px - 0.36em)`,
              fontSize: '0.72rem', whiteSpace: 'nowrap', pointerEvents: 'none',
            }}
          >
            ▲ the book's rule of thumb, p. 10
          </span>
        </label>
        <FractionSlider p={p} setP={setP} />
        <button className="quiet" onClick={reset} style={{ alignSelf: 'center' }}>
          Back to the defaults
        </button>
      </div>
      <div className="cols plot">
        <CapsPlot p={p} threads={threads} rate={rate} cap={cap} />
        <div className="panel" data-readout>
          <p className="muted" style={{ margin: 0 }}>Kernel speed-up, s = min(n × r, M)</p>
          <p style={{ margin: '0 0 0.6rem' }} data-out="kernel">
            min({fmt(threads * rate)}, {fmt(cap)}) = <b>{times(now.s)}</b>, set by{' '}
            {now.limitedBy === 'memory' ? <span className="chip global">memory</span> : <b>the thread count</b>}
          </p>
          <p className="muted" style={{ margin: 0 }}>Overall speed-up</p>
          <p style={{ fontSize: '1.9rem', fontWeight: 600, margin: '0 0 0.4rem', lineHeight: 1.15 }} data-out="overall">{times(now.overall)}</p>
          {now.s < 1 ? (
            <p style={{ margin: '0 0 0.5rem' }} data-out="slower">
              With this few threads the kernel is slower than the CPU core it replaces.
            </p>
          ) : null}
          <p style={{ margin: '0 0 0.6rem' }} data-out="detail">
            Ceiling from the untouched part: {times(top)}. The untouched part is {share(now.untouchedShare)} of the time
            left. Break-even at {count(c.breakEven)} threads; memory saturates at {count(c.saturation)}.
          </p>
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th>One change from here</th>
                <th className="num">Overall</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {lever('threads', '10× the threads', lv.threads)}
              {lever('memory', '10× the memory cap', lv.memory)}
              {lever('untouched', '1/10 of the untouched part', lv.untouched)}
            </tbody>
          </table>
          <p style={{ margin: '0.7rem 0 0' }} className="row" data-goals>
            {goal(50)}
            {goal(100)}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- classification
type Challenge = 'work' | 'memory' | 'input' | 'sync';
const CHALLENGES: { id: Challenge; label: string }[] = [
  { id: 'work', label: 'Algorithm and work' },
  { id: 'memory', label: 'Memory-bound' },
  { id: 'input', label: 'Input sensitivity' },
  { id: 'sync', label: 'Synchronisation' },
];

interface Card {
  id: string;
  text: ReactNode;
  answer: Challenge;
  /** Shown once the card is placed: why, and where the course deals with it. */
  done: ReactNode;
  /** Shown for each wrong choice. */
  why: Partial<Record<Challenge, ReactNode>>;
}

// All eight situations are ours, not the book's. Each is justified from the notes of
// the chapter it names; the comment above a card says where.
const CARDS: Card[] = [
  // ch. 1 p. 11: recurrences need reformulating and prefix sum is the standard tool.
  // notes/ch11.md, "Relevance": top-p as an inclusive scan over sorted probabilities (our mapping).
  {
    id: 'topp',
    text: (
      <>
        Top-p sampling needs the running sum of the sorted probabilities. Written as a loop, each partial sum is the
        one before it plus one more term.
      </>
    ),
    answer: 'work',
    done: (
      <>
        A recurrence: step i needs step i − 1, so as written there is nothing for a second thread to do.{' '}
        <a href={href('ch11')}>Chapter 11</a> reformulates it as a parallel prefix sum, and module X1 uses that for
        top-p.
      </>
    ),
    why: {
      memory: <>One row of probabilities is not much to fetch. The obstacle is that each sum seems to need the one before it.</>,
      input: <>Every row has the same chain of dependences, whatever values it holds.</>,
      sync: <>Waiting is what some solutions cost. The difficulty comes first: as written, no two steps of the loop can run at once.</>,
    },
  },
  // ch. 5 §5.1 (pp. 94–97), the FLOP/B count applied to a matrix-vector product;
  // ch. 20 §20.6 (p. 505) reaches the same conclusion for decoding. See ch05/5.1/q-decode.
  {
    id: 'matvec',
    text: (
      <>
        Generating one token at batch size 1: each weight matrix multiplies a single vector, so every weight is fetched
        once and used once.
      </>
    ),
    answer: 'memory',
    done: (
      <>
        Two operations per weight fetched, however the kernel is written, so the speed is set by how fast weights
        arrive. <a href={href('ch05', '5.1')}>Chapter 5, §5.1</a> gives the count; batching and quantisation are the
        ways out.
      </>
    ),
    why: {
      work: <>The parallel kernel performs the same multiplications and additions as the loop, no more.</>,
      input: <>Dense weights: every thread gets the same amount of work.</>,
      sync: <>Each output element belongs to one thread. Nobody waits for anybody.</>,
    },
  },
  // ch. 20 §20.6 (p. 506): unequal sequence lengths may unbalance work across thread
  // blocks and under-use the GPU; in-flight batching is the serving-side answer.
  {
    id: 'lengths',
    text: <>One launch serves a batch of three prompts that are 12, 900 and 4,000 tokens long.</>,
    answer: 'input',
    done: (
      <>
        Unequal lengths give thread blocks unequal work and leave part of the GPU idle. The book notes this for LLM
        batches in <a href={href('ch20')}>chapter 20</a> (p. 506).
      </>
    ),
    why: {
      work: <>Nothing about the algorithm changed. The same kernel on three equal prompts would be balanced.</>,
      memory: <>Perhaps that too, but it would be just as true of three equal prompts. What is special here is the mix of lengths.</>,
      sync: <>No thread needs another's result. The idle time comes from uneven work, not from waiting at a barrier.</>,
    },
  },
  // ch. 9 §9.2–9.3 (pp. 206, 209): atomic updates to one location are serialised.
  // notes/ch09.md, "Relevance": token-count histograms for repetition penalties (our mapping).
  {
    id: 'counters',
    text: (
      <>
        A repetition penalty needs a count for every token id seen so far. One thread per context token increments the
        counter for its token, and a frequent token is hit by many threads.
      </>
    ),
    answer: 'sync',
    done: (
      <>
        The updates must be atomic, and atomic updates to one location run one after another while the other threads
        wait. <a href={href('ch09')}>Chapter 9</a> is about this cost and how to reduce it.
      </>
    ),
    why: {
      work: <>Counting N tokens takes N increments in parallel as in sequence. The cost is in making them safe.</>,
      memory: <>The counters are small. What slows the threads is taking turns, not fetching.</>,
      input: <>The token distribution decides how bad it gets, and chapter 9 says as much. The cost itself is threads taking turns on a counter.</>,
    },
  },
  // ch. 11 §11.5 (pp. 265–266): Kogge-Stone performs N·log2N − (N − 1) additions against
  // a minimum of N − 1; for 512 elements, between 8 and 9 times the sequential work.
  {
    id: 'worksum',
    text: (
      <>
        A parallel prefix sum over N values finishes in log<sub>2</sub> N steps. It performs about N × log<sub>2</sub> N
        additions, where the sequential loop performs N − 1.
      </>
    ),
    answer: 'work',
    done: (
      <>
        More total work than the sequential code: for 512 values, between 8 and 9 times as much (pp. 265–266).{' '}
        <a href={href('ch11')}>Chapter 11</a> calls the measure work efficiency, then repairs it.
      </>
    ),
    why: {
      memory: <>Count the additions, not the bytes. The card is about how much arithmetic there is in total.</>,
      input: <>The count depends on N only, not on the values.</>,
      sync: <>The steps do need barriers between them. But the number in the card is additions performed, and it is larger than the loop's.</>,
    },
  },
  // ch. 17 p. 401 (irregularity costs: load imbalance, control divergence) and §17.3
  // pp. 408–410 (CSR: one thread per row; the trip count is the row length).
  {
    id: 'sparse',
    text: <>A sparse matrix has between 1 and 10,000 non-zeros per row. The kernel gives each row to one thread.</>,
    answer: 'input',
    done: (
      <>
        Each thread loops over its own row, so neighbouring threads do very different amounts of work.{' '}
        <a href={href('ch17')}>Chapter 17</a> names the cost (load imbalance and control divergence, pp. 401 and 410)
        and makes the data more regular to reduce it.
      </>
    ),
    why: {
      work: <>Each non-zero is multiplied once, as in the sequential loop.</>,
      memory: <>Sparse kernels are short of bandwidth too, but that would hold with equal rows. The spread from 1 to 10,000 is the point here.</>,
      sync: <>Each row's result belongs to one thread, so there are no atomic updates and no barriers.</>,
    },
  },
  // ch. 5 §5.4 (pp. 108–112): the tiled kernel and its two __syncthreads().
  {
    id: 'barrier',
    text: (
      <>
        In the tiled matrix multiplication kernel, every thread of a block loads one element into the block's tile,
        then stops until all the others have loaded theirs.
      </>
    ),
    answer: 'sync',
    done: (
      <>
        A barrier: threads that arrive early wait instead of working. <a href={href('ch05', '5.4')}>Chapter 5, §5.4</a>{' '}
        shows why the kernel needs two of them.
      </>
    ),
    why: {
      work: <>The tiled kernel performs the same multiply-adds as the naive one.</>,
      memory: <>Tiling is the cure for the memory problem. The stop is the price it pays, and the price has its own name.</>,
      input: <>Dense matrices: every thread does the same thing.</>,
    },
  },
  // ch. 5 §5.1 (p. 96): vector addition does 1 FLOP per 12 bytes, 0.083 FLOP/B, deeply
  // memory-bound; ch. 1 p. 12 for "embarrassingly parallel".
  {
    id: 'bias',
    text: (
      <>
        Adding a bias to every element of a large tensor. No thread needs anything from any other thread. One of the
        four still applies.
      </>
    ),
    answer: 'memory',
    done: (
      <>
        Like the book's vector addition, which performs one operation for every 12 bytes it moves (
        <a href={href('ch05', '5.1')}>chapter 5, §5.1</a>, p. 96): embarrassingly parallel, and held back by memory all
        the same.
      </>
    ),
    why: {
      work: <>One addition per element, in parallel as in sequence.</>,
      input: <>Every element costs the same.</>,
      sync: <>There is none to pay. The book's term for that is embarrassingly parallel, and it is why something else sets the speed.</>,
    },
  },
];

function WhichChallenge() {
  const [picks, setPicks] = useState<Record<string, Challenge | undefined>>({});
  const placed = CARDS.filter((c) => picks[c.id] === c.answer).length;
  return (
    <Bench
      title="Which challenge is it?"
      task="Eight situations. Give each the one of the four challenges that fits it best."
      foot="The situations are ours, not the book's. The feedback names the chapter where this course deals with each."
    >
      <div data-which>
        <div className="panel">
          <div className="row" style={{ justifyContent: 'space-between', marginBottom: '0.2rem' }}>
            <b>Eight situations</b>
            <span className="muted" data-which-score>
              {placed} of {CARDS.length} placed
            </span>
          </div>
          {CARDS.map((c, n) => {
            const pick = picks[c.id];
            const done = pick === c.answer;
            return (
              <div
                key={c.id}
                data-card={c.id}
                data-solved={done ? 'true' : 'false'}
                style={{ padding: '0.75rem 0', borderTop: n ? '1px solid var(--rule-soft)' : undefined }}
              >
                <p style={{ margin: '0 0 0.5rem', maxWidth: '46rem', fontSize: '0.92rem' }}>
                  <span className="muted">{n + 1}.</span> {c.text}
                </p>
                <div className="row" style={{ gap: '0.4rem' }} role="group" aria-label={`Challenge for situation ${n + 1}`}>
                  {CHALLENGES.map((ch) => (
                    <button
                      key={ch.id}
                      className="quiet"
                      data-choice={ch.id}
                      aria-pressed={pick === ch.id}
                      disabled={done && ch.id !== c.answer}
                      onClick={() => (done ? undefined : setPicks((s) => ({ ...s, [c.id]: ch.id })))}
                    >
                      {ch.label}
                    </button>
                  ))}
                </div>
                {pick ? (
                  <p data-feedback={c.id} role="status" style={{ margin: '0.5rem 0 0', maxWidth: '46rem' }}>
                    {done ? (
                      <>
                        <span className="chip ok">placed</span> {c.done}
                      </>
                    ) : (
                      <>
                        <span className="chip fail">not this one</span> {c.why[pick]}
                      </>
                    )}
                  </p>
                ) : null}
              </div>
            );
          })}
        </div>
        <div className="row" style={{ marginTop: '0.8rem', justifyContent: 'flex-end' }}>
          <button className="quiet" onClick={() => setPicks({})}>
            Start again
          </button>
        </div>
        {placed === CARDS.length ? (
          <div className="panel" style={{ marginTop: '0.8rem' }} data-which-done>
            <p style={{ margin: 0 }}>
              <b>All eight placed.</b> The four names are a working vocabulary for the kernels you will write: say
              which one a kernel suffers from before you decide what to change. The last card is the pattern to
              remember. A kernel can be easy on three counts and hard on the fourth.
            </p>
          </div>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questions13: Record<string, Question> = {
  'ch01/1.3/q-fivehundred': {
    kind: 'numeric',
    id: 'ch01/1.3/q-fivehundred',
    prompt: (
      <>
        No real kernel is infinitely fast. Say you make the parallel part {fmt(FIVE_HUNDRED.s)} times faster. What
        parallel fraction p gives about {FIVE_HUNDRED.target}× overall? Give it to one decimal place.
      </>
    ),
    answer: Number((100 * fractionNeeded(FIVE_HUNDRED.target, FIVE_HUNDRED.s)!).toFixed(1)),
    tol: 0.0001,
    unit: '%',
    explain: (
      <p>
        {FIVE_HUNDRED.target}× overall means the run takes 1/{FIVE_HUNDRED.target} = {fmt(100 / FIVE_HUNDRED.target)}%
        of its old time. The accelerated part, {fmt(FIVE_HUNDRED.s)} times faster, takes about p /{' '}
        {fmt(FIVE_HUNDRED.s)}, close to 0.1%. That leaves about 0.1% for the part left alone, so p is about 99.9%
        (to four places, {frac(fractionNeeded(FIVE_HUNDRED.target, FIVE_HUNDRED.s)!)}). The untouched 0.1% now takes
        as long as all the accelerated work, as the untouched 1% did in your second prediction. So the 99% of the
        task makes 100× possible only with a part of unbounded speed. With a real, finite speed-up, results of
        several hundred times need the "more than 99.9%" the book quotes. Set the Amdahl explorer to 99.9% and{' '}
        {times(FIVE_HUNDRED.s)}: it shows {times(speedup(0.999, FIVE_HUNDRED.s))}.
      </p>
    ),
    source: "our numbers; the law and the 99.9% figure are the book's, ch. 1 §1.3, pp. 9–10",
  },
  'ch01/1.3/q-kernelneeded': {
    kind: 'numeric',
    id: 'ch01/1.3/q-kernelneeded',
    prompt: (
      <>
        A profile shows that kernels take 96% of the time of a decode step. You need the step to run 20 times faster
        end to end, and you can only change the kernels. How many times faster must they become?
      </>
    ),
    answer: 96,
    tol: 0.01,
    unit: '×',
    explain: (
      <p>
        A 20× step may take 1/20 = 0.05 of the original time. The untouched 4% uses 0.04 of that, which leaves 0.01 for
        the kernels: 0.96 / 0.01 = 96×. Had the kernels been 95% of the step, the untouched 5% would use the whole
        budget. 20× would then be the ceiling itself, and no finite kernel speed-up reaches it.
      </p>
    ),
    source: "our numbers; the law is the book's, ch. 1 §1.3, p. 9",
  },
  'ch01/1.3/q-hostshare': {
    kind: 'numeric',
    id: 'ch01/1.3/q-hostshare',
    prompt: (
      <>
        In a decode step, sampling and the copies that feed it run on the host and take 8% of the time. Everything else
        is kernels. However fast you make the kernels, what is the most the step can gain end to end?
      </>
    ),
    answer: 12.5,
    tol: 0.005,
    unit: '×',
    explain: (
      <p>
        The ceiling is 1 / (1 − p) with p = 0.92: 1 / 0.08 = 12.5×. Past that, the only move left is to shrink the 8%
        itself. That is the case for running the sampler on the GPU, which module X1 does. In llama.cpp the default
        sampler chain runs on the host.
      </p>
    ),
    source: "our framing; the law is the book's, ch. 1 §1.3, p. 9; llama.cpp checked 2026-11-20 (notes/today/ch01-facts.md, D2)",
  },
  'ch01/1.3/q-bigslice': {
    kind: 'choice',
    id: 'ch01/1.3/q-bigslice',
    prompt: (
      <>
        You have time for one of two jobs. Option A makes a kernel that takes 60% of the run time 10 times faster.
        Option B makes a kernel that takes 30% of the run time infinitely fast. Which gives the larger overall
        speed-up?
      </>
    ),
    options: [
      { label: 'A.', correct: true },
      {
        label: 'B.',
        why: 'Infinite beats ten on the part, but the part is only 30% of the time. The other 70% caps the whole at 1.43×.',
      },
      { label: 'They come out the same.', why: 'A leaves 40% + 6% = 46% of the time. B leaves 70%.' },
    ],
    explain: (
      <p>
        A: 1 / (0.4 + 0.6 / 10) = {fmt(speedup(0.6, 10))}×. B: 1 / 0.7 = {fmt(speedup(0.3, Infinity))}×. The size of
        the slice counts for more than how fast you make it. Profile first, then choose the kernel.
      </p>
    ),
    source: "our numbers; the law is the book's, ch. 1 §1.3, p. 9",
  },
  'ch01/1.3/q-saturated': {
    kind: 'choice',
    id: 'ch01/1.3/q-saturated',
    prompt: (
      <>
        A straightforward port of a kernel already keeps memory saturated. You rewrite its arithmetic so that each
        element needs half as many instructions, and you change nothing about what it reads and writes. What happens to
        its run time?
      </>
    ),
    options: [
      {
        label: 'It roughly halves.',
        why: 'That would be true of a compute-bound kernel. The threads of this one were already waiting for data.',
      },
      { label: 'Almost nothing.', correct: true },
      {
        label: 'It gets slower.',
        why: 'Nothing was added. The same bytes move at the same rate; the arithmetic finishes sooner and waits longer.',
      },
    ],
    explain: (
      <p>
        The kernel is memory-bound: its speed is set by how fast bytes arrive, and the same bytes still have to move.
        What helps is fewer bytes per operation, either by reusing data from on-chip memory (chapters 5, 6 and 15) or
        by making each value smaller (quantisation, module X2).
      </p>
    ),
    source: 'ch. 1 §1.3–1.4, pp. 10–12; chapter 5 §5.1 gives the method',
  },
};
// The same record under the name the file-based convention gives it.
export const questionsS13 = questions13;

// ---------------------------------------------------------------- the page
export function S13() {
  const [p, setP] = useState(DEFAULT_P);
  const left30 = remaining(book30.p, book30.s);
  const left99 = remaining(book99.p, book99.s);
  const atDefaults = limits(DEFAULT_P, DEFAULTS.threads, DEFAULTS.rate, DEFAULTS.cap);
  return (
    <>
      <p>
        Before you write a kernel, work out how much the whole program can gain from it. This page gives you the
        arithmetic for that, and then the four difficulties that decide how hard the gain is to get. Chapter 1 of the
        book has no code and no exercises, so its own worked cases serve as the predictions here.
      </p>
      <Source section="1.3" pages="9–11" />

      <h2>Speed-up</h2>
      <p>
        The <b>speed-up</b> of system A over system B is the time B takes divided by the time A takes. The order
        matters: a figure below 1 means A is the slower one. The book's example: 200 s on B and 10 s on A is a
        speed-up of 200 / 10 = 20× for A over B.
      </p>
      <Source section="1.3" pages="9" />
      <p>
        The figure says as much about the baseline as about the port: how well the CPU suits the task, and how good
        the CPU version was. So the book asks you to give the CPU a fair chance. Most applications have parts that run
        better there, and GPU code is meant to complement them.
      </p>
      <Source section="1.3" pages="10" />

      <h2>Two predictions</h2>
      <p>
        A kernel is never the whole program. Commit to a number for each of these before you read on. No formula is
        needed: think in run time, part by part.
      </p>
      <Source section="1.3" pages="9" />
      <Predict
        id="ch01/1.3/p-thirty"
        prompt={
          <>
            A program spends 30% of its run time in a part you can parallelise. You make that part 100 times faster and
            leave the rest alone. What is the overall speed-up?
          </>
        }
        unit="×"
      >
        {(v) => (
          <p>
            The accelerated part shrinks from 30% of the time to {share(left30.accelerated)}. The other{' '}
            {share(left30.untouched)} is untouched, so the run takes {share(left30.total)} of what it did: 1 /{' '}
            {fmt(left30.total)} = <b>{times(speedup(book30.p, book30.s))}</b>.{' '}
            {v >= 1.395 && v < 1.4275
              ? 'That is what you had.'
              : v >= 1.4275 && v <= 1.44
                ? 'You dropped the 0.3% that is left of the accelerated part. That is the next case.'
                : v >= 1.25 && v <= 1.35
                  ? 'You have the saving right: about 30% of the time. But a run that takes 70.3% as long is 1 / 0.703 times as fast, not 1.3 times.'
                  : v >= 20
                    ? 'If you were near 30 or 100: the other 70% did not move, and it is now nearly all of the run.'
                    : 'Work in time, not in speed: add up what is left of each part, then divide.'}{' '}
            Make the same part infinitely fast and the run still takes 70% of the time: no more than{' '}
            {times(ceiling(book30.p))}. Both figures are the book's.
          </p>
        )}
      </Predict>
      <Predict
        id="ch01/1.3/p-ninetynine"
        prompt={<>Now the part is 99% of the run time, and again you make it 100 times faster. What is the overall speed-up?</>}
        unit="×"
      >
        {(v) => (
          <p>
            The 99% shrinks to {share(left99.accelerated)}. The untouched 1% is still 1%. The run takes{' '}
            {share(left99.total)} of what it did: 1 / {fmt(left99.total)} = <b>{times(speedup(book99.p, book99.s))}</b>
            ; the book says about 50.{' '}
            {Math.abs(v - speedup(book99.p, book99.s)) <= 1.5
              ? 'That is what you had.'
              : v >= 90
                ? 'Nearly everything was accelerated, so 99 or 100 is the natural guess. It forgets that the untouched 1% is now as large as all the accelerated work.'
                : 'Same method as before: 1% untouched, plus 0.99% left of the accelerated part.'}{' '}
            One percent of the program has taken half of the hundred.
          </p>
        )}
      </Predict>

      <h2>Amdahl's law</h2>
      <p>
        What you just worked out is Amdahl's law: the speed-up of a whole program is limited by the share of its time
        that you leave alone. The book states it in words and shows it through those cases. Written out, with p the
        fraction of the original run time in the part you speed up and s the speed-up of that part, the overall figure
        is 1 / ((1 − p) + p / s).
      </p>
      <Beyond title="Our formula">
        <p>
          The book prints no formula, only numbers (p. 9). This one is ours, and the book's three cases match it: 1.42,
          1.43 and about 50.
        </p>
      </Beyond>
      <p>
        The explorer starts on the case you just predicted. Use the presets to check the other two, then work on the
        task in its header.
      </p>
      <AmdahlBench p={p} setP={setP} />
      <p>
        Three things to notice. The presets give {times(speedup(0.3, 100))}, {times(speedup(0.3, Infinity))} and{' '}
        {times(speedup(0.99, 100))}, the book's figures. The ceiling depends on p alone: nothing you do to s moves the
        dashed line. And close to the ceiling, effort stops paying. At p = 99%, taking the part from 100× to 1,000×
        moves the whole only from about {fmt(speedup(0.99, 100), 2)}× to about {fmt(speedup(0.99, 1000), 2)}×, because
        by then most of the time that is left belongs to the part you did not touch.
      </p>
      <Simplification>
        <p>
          Two parts only, and they do not overlap in time; chapter 22 (
          <a href={href('ch22', '22.3')}>§22.3</a>) relaxes that. The untouched part is
          assumed to stay the same size, although moving work to a GPU adds copies and kernel launches to it. The
          problem size is fixed.
        </p>
      </Simplification>
      <p>
        This is the background to a remark the book makes right after its examples. Speed-ups above 100× have been
        reported, it says, but typically only after optimisation and algorithm changes that put more than 99.9% of the
        work in the parallel part. Why 99.9% and not the 99% of the task? The task was the first half of the reason:
        it assumed a part of unbounded speed. The next question is the second half.
      </p>
      <Source section="1.3" pages="9–10" />
      <Ask q={questions13['ch01/1.3/q-fivehundred']} />
      <Ask q={questions13['ch01/1.3/q-kernelneeded']} />

      <h2>The same sum for a decode step</h2>
      <p>
        Now apply it to what you build. Generating one token is a chain of kernels (the matrix products, attention,
        normalisation) plus work on the host: the tokeniser, the sampler chain, and copying the logits back so that
        the sampler can read them. If you speed up only the kernels, the host work is the 1 − p of the law. In
        llama.cpp the default sampler chain runs on the host over an array of candidates, so by default it belongs to
        the part that does not speed up.
      </p>
      <Beyond>
        <p>
          The mapping to a decode step is ours. The book works the same law on a molecular-dynamics application in{' '}
          <a href={href('ch22', '22.3')}>§22.3</a> (pp. 537–538), where the modules left on the host cap the whole.
        </p>
      </Beyond>
      <p>
        That is a default, not a law. llama.cpp also has an experimental interface that runs samplers inside the
        backend, beside the kernels, and <a href={href('x1')}>module X1</a> of this course has you write that kind of
        kernel: greedy, top-k and top-p selection on the GPU. The two questions below are the reason it is worth
        doing.
      </p>
      <Today checked="2026-11-20 against llama.cpp's llama.h (notes/today/ch01-facts.md, D2)">
        <p>
          The header marks the backend sampling interface as experimental. We did not trace which samplers implement
          it, or whether any front end turns it on.
        </p>
      </Today>
      <Ask q={questions13['ch01/1.3/q-hostshare']} />
      <Ask q={questions13['ch01/1.3/q-bigslice']} />

      <h2>Where the first 10× comes from, and where it stops</h2>
      <p>
        The book gives three rules of thumb, all from its authors' experience. The first two: a good GPU implementation
        of a suitable application can run more than 100× faster than sequential code on one CPU core, and where the
        application has data parallelism, 10× is often reached with a few hours of work.
      </p>
      <Source section="1.2" pages="7" />
      <p>
        The third is the one this course is built around. A straightforward port often saturates DRAM bandwidth and
        stops at about 10×. Going further means changing the code so that data is reused from the GPU's on-chip
        memories and far fewer accesses go to DRAM, and then working around how small those memories are. In this
        course that is chapters 5, 6 and 15. We found no vendor source for any of the three figures; take each as the
        book's.
      </p>
      <Source section="1.3" pages="10" />
      <p>
        Earlier in the chapter the book also says that graphics chips have run at roughly 10× the memory bandwidth of
        the CPUs of their day (p. 5). It states both tens and does not link them. You do not have to take either
        on trust. <a href={href('ch05', '5.1')}>Chapter 5</a> turns the remark about saturation into a calculation: its
        roofline gives a kernel's ceiling from its operations per byte and the device's bandwidth, and lab 5.1
        measures the bandwidth of your own device. Chapter 5 also shows a case where the model and a measurement
        disagree by a wide margin, and why (<a href={href('ch05', '5.1')}>"When a measurement lands above the
        roof"</a>). So the course has you predict and then measure, and this rule of thumb is no exception.
      </p>
      <Today checked="2026-11-20 against the CUDA programming guide and vendor specification pages (notes/today/ch01-facts.md, A8)">
        <p>
          NVIDIA's guide confirms the direction: more memory bandwidth than a CPU in a similar price and power
          envelope. On the vendor figures we checked, the ratio for one data-centre GPU against one server CPU socket
          is below ten. On Apple silicon one memory serves both processors, so there is no ratio.
        </p>
      </Today>
      <p>
        The explorer below puts three limits in one place: how many threads run, how fast memory can feed them, and how
        much of the program you left alone. The book gives each of them in words; the model that combines them is
        ours. It shares its parallel fraction with the explorer above.
      </p>
      <CapsBench p={p} setP={setP} />
      <p>
        Three things to notice. Past the point where memory saturates, more threads buy nothing: with a memory cap of{' '}
        {DEFAULTS.cap} and 99% of the time in the kernel, the result is {times(atDefaults.overall)} however many
        threads run. Raising the cap is the only change that moves it much: to{' '}
        {times(limits(DEFAULT_P, DEFAULTS.threads, DEFAULTS.rate, 100).overall)} at a cap of 100. In a real kernel that means fewer bytes per operation, through reuse (chapters 5, 6 and 15) or
        through smaller values (quantisation). After that the kernel is no longer what holds you back, and only a
        larger parallel fraction helps. For a decode step, that means moving host work such as sampling onto the GPU.
      </p>
      <Simplification>
        <p>
          Three hard limits with sharp corners, where real curves bend. The memory cap is one number here; chapter 5
          computes it from a kernel's operations per byte and a device's bandwidth, and caches change it. The rate of
          a thread is a constant, and threads scale perfectly until memory saturates. The book states the 10× figure
          and does not derive it.
        </p>
      </Simplification>
      <Ask q={questions13['ch01/1.3/q-saturated']} />
      <p>
        The book pictures all of this as a peach. The pit is the sequential part, which can be a large share of the
        code and is usually a small share of the run time; the flesh is the data-parallel part, where the time is. The
        time bars of the first explorer are our picture of the same thing.
      </p>
      <Source section="1.3" pages="10–11" figs="Fig. 1.2" />

      <h2>Four difficulties the book keeps returning to</h2>
      <p>
        Parallel programming is easy, the book remarks, as long as you do not care about performance. It names four
        things that make the fast version hard. The rest of the book, and of this course, is largely a set of answers
        to them.
      </p>
      <Source section="1.4" pages="11–12" />
      <p>
        <b>Algorithmic complexity.</b> A parallel algorithm can do more total work than the sequential one it replaces,
        sometimes enough to lose on large inputs. A problem that is naturally a recurrence, where each step needs the
        result of the one before, has to be reformulated before a second thread has anything to do, and the new form
        may add redundant work. The standard tool is the prefix sum, and the measure of the extra work is{' '}
        <b>work efficiency</b>. Both arrive in chapter 11.
      </p>
      <p>
        <b>Memory-bound execution.</b> A program whose speed is set by how fast its data arrives is <b>memory-bound</b>
        ; one whose speed is set by its arithmetic is <b>compute-bound</b>. For the first kind, faster arithmetic buys
        nothing. Chapters 5 and 6 are about getting more operations out of each byte.
      </p>
      <p>
        <b>Sensitivity to input data.</b> Parallel programs react more than sequential ones to the shape of their
        input. Erratic sizes and uneven distributions hand the threads uneven amounts of work. The remedies are to
        make the data more regular or to fit the assignment of data to threads, and the book places them in its
        pattern and application chapters.
      </p>
      <p>
        <b>Synchronisation overhead.</b> Work whose threads need almost nothing from each other is called{' '}
        <b>embarrassingly parallel</b>. Everything else needs synchronisation operations such as barriers and atomic
        operations, and each of them makes threads wait instead of work. Cutting that waiting is a theme of the whole
        book.
      </p>
      <table className="data wide" data-challenges style={{ maxWidth: '62rem' }}>
        <thead>
          <tr>
            <th style={{ minWidth: '8.5rem' }}>Challenge</th>
            <th style={{ minWidth: '9.5rem' }}>The book's term</th>
            <th style={{ minWidth: '12rem' }}>Where the book meets it</th>
            <th style={{ minWidth: '11rem' }}>In this course's own modules</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>The parallel version does more work</td>
            <td>algorithmic complexity; work efficiency</td>
            <td>ch. 11 (named on p. 11)</td>
            <td>X1 sampling: top-p needs a prefix sum</td>
          </tr>
          <tr>
            <td>Memory sets the speed</td>
            <td>memory-bound, against compute-bound</td>
            <td>chs. 5 and 6 (named on pp. 11–12); ch. 15 is our addition</td>
            <td>X2 quantised kernels: fewer bytes per operation</td>
          </tr>
          <tr>
            <td>Threads get uneven work</td>
            <td>sensitivity to input data</td>
            <td>its pattern and application chapters (p. 12); chs. 13, 17 and 18 are our reading</td>
            <td>neither module is built around it</td>
          </tr>
          <tr>
            <td>Threads wait for each other</td>
            <td>synchronisation overhead; embarrassingly parallel</td>
            <td>a theme throughout (p. 12); chs. 5, 9, 10 and 11 are our reading</td>
            <td>X1 sampling: counters that many threads update</td>
          </tr>
        </tbody>
      </table>
      <p>
        Where the table gives a page, the chapter number is the book's own. The rest of the mapping is ours, made from
        our notes on those chapters, and so is the last column: the book has no sampling or quantisation chapter.
      </p>

      <h3>Put the names to work</h3>
      <p>
        Each situation below is one you will meet when you write inference kernels. Decide which of the four it is
        mainly an instance of. Several are easy on most counts, which is what makes the remaining one worth naming.
      </p>
      <WhichChallenge />
      <p>
        Next, <a href={href('ch01', '1.7')}>the book, this edition, and your path through it</a>: which chapters carry
        these four through to matrix multiplication, attention and sampling.
      </p>
    </>
  );
}
