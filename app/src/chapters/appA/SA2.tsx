import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { href } from '../../lib/router';
import {
  BF16, E2M1, E4M3, E5M2, FP16, FP32,
  anatomy, binade, bitString, classify, decode, fromFields, makeFormat, maxFinite, minNormal, minSubnormal,
  round, show, topCode, ulp, values,
  type Classified, type Format, type Subnormals,
} from '../../sim/minifloat';

// ---------------------------------------------------------------- small pieces
const minus = (n: number | string) => String(n).replace(/-/g, '−');
function parseValue(t: string): number | null {
  const s = t.trim().toLowerCase().replace(/[−–]/g, '-').replace(/[,_\s]/g, '');
  if (s === '') return null;
  const p = /^2\^(-?\d+(\.\d+)?)$/.exec(s);
  if (p) return 2 ** Number(p[1]);
  const v = Number(s);
  return Number.isNaN(v) ? null : v;
}
const P_GAP = 'appA/A.2/p-gap';

/** Width of a container in px, so a drawing can be laid out for it instead of scaled. */
function useWidth(initial = 900) {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(initial);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver((entries) => setW(Math.round(entries[0].contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, Math.max(300, Math.min(900, w))] as const;
}

// ---------------------------------------------------------------- explorer: the number line (Figs. A.4–A.7)
const CONVS: { id: Subnormals; label: string; short: string; fig: string }[] = [
  { id: 'nozero', label: 'no zero', short: 'no zero', fig: 'Fig. A.5' },
  { id: 'abrupt', label: 'abrupt underflow', short: 'abrupt', fig: 'Fig. A.6' },
  { id: 'denorm', label: 'denormals', short: 'denorm', fig: 'Fig. A.7' },
];

function Lines({
  fmts,
  conv,
  lo,
  hi,
  probe,
  showProbe,
  label,
}: {
  fmts: Record<Subnormals, Format>;
  conv: Subnormals;
  lo: number;
  hi: number;
  probe: number;
  /** False until the prediction above is committed: the rings would give its answer away. */
  showProbe: boolean;
  label: string;
}) {
  const [ref, W] = useWidth();
  const narrow = W < 560;
  const L0 = narrow ? 64 : 132;
  const R0 = W - 18;
  const rowH = 44;
  const top = 18;
  const H = top + rowH * 3 + 30;
  const X = (v: number) => L0 + ((v - lo) / (hi - lo)) * (R0 - L0);
  const f0 = fmts.denorm;
  // Ticks: 0 and the powers of two that start intervals, skipping labels that would collide.
  const ticks: number[] = [0];
  for (let k = Math.round(Math.log2(minNormal(fmts.nozero))); 2 ** k <= hi; k++) if (2 ** k >= lo) ticks.push(2 ** k);
  if (hi <= 8) for (let i = 3; i <= hi; i += 2) ticks.push(i);
  ticks.sort((p, q) => p - q);
  const shown: number[] = [];
  for (const t of ticks) if (!shown.length || X(t) - X(shown[shown.length - 1]) >= 30) shown.push(t);
  const probeIn = showProbe && probe >= lo && probe <= hi;
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={label} style={{ maxWidth: '100%', height: 'auto', display: 'block' }}>
        {CONVS.map((c, r) => {
          const f = fmts[c.id];
          const y = top + rowH * r + rowH / 2;
          const pos = values(f).filter((v) => v >= 0);
          const inView = pos.filter((v) => v >= lo && v <= hi);
          const sel = c.id === conv;
          const smallest = pos.find((v) => v > 0);
          const near = round(f, probe);
          return (
            <g key={c.id} data-conv={c.id} data-count={pos.filter((v) => v > 0).length} opacity={sel ? 1 : 0.55}>
              <text x={4} y={y + 4} fontSize="12" fontWeight={sel ? 650 : 400}>
                {narrow ? c.short : c.label}
              </text>
              {smallest !== undefined && smallest > lo ? (
                <rect x={X(Math.max(lo, 0))} y={y - 7} width={Math.max(0, X(Math.min(smallest, hi)) - X(Math.max(lo, 0)))} height={14}
                  style={{ fill: 'var(--rule-soft)', stroke: 'var(--rule)' }} strokeDasharray="3 2" />
              ) : null}
              <line x1={L0} x2={R0} y1={y} y2={y} style={{ stroke: 'var(--ink-3)' }} />
              {inView.map((v) => (
                <circle key={v} cx={X(v)} cy={y} r={sel ? 3.6 : 3} style={{ fill: 'var(--ink)' }} />
              ))}
              {!pos.includes(0) && lo <= 0 ? (
                <text x={X(0)} y={y + 4} fontSize="12" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>×</text>
              ) : null}
              {probeIn && near >= lo && near <= hi ? (
                <circle cx={X(near)} cy={y} r={7} style={{ fill: 'none', stroke: 'var(--ink)' }} strokeWidth={1.6} />
              ) : null}
            </g>
          );
        })}
        <line x1={L0} x2={R0} y1={H - 26} y2={H - 26} style={{ stroke: 'var(--rule)' }} />
        {shown.map((t) => (
          <g key={t}>
            <line x1={X(t)} x2={X(t)} y1={H - 29} y2={H - 23} style={{ stroke: 'var(--ink-3)' }} />
            <text x={X(t)} y={H - 9} fontSize="11" textAnchor="middle">
              {t === 0 || t >= 1 / 16 ? String(t) : `2^${minus(Math.log2(t))}`}
            </text>
          </g>
        ))}
        {probeIn ? (
          <g>
            <line x1={X(probe)} x2={X(probe)} y1={top - 4} y2={H - 26} style={{ stroke: 'var(--ink-2)' }} strokeDasharray="4 3" />
            <text x={X(probe) + 4} y={top + 6} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
              x
            </text>
          </g>
        ) : null}
        <text x={R0} y={top - 2} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
          {f0.name} format, positive values (negatives mirror them)
        </text>
      </svg>
    </div>
  );
}

function NumberLineBench() {
  const { progress } = useProgress();
  const revealed = progress.answers[P_GAP] != null;
  const [e, setE] = useState(2);
  const [m, setM] = useState(2);
  const [conv, setConv] = useState<Subnormals>('nozero');
  const [probe, setProbe] = useState(0.2);
  const [text, setText] = useState('0.2');
  const fmts = useMemo(
    () => ({ nozero: makeFormat(e, m, 'nozero'), abrupt: makeFormat(e, m, 'abrupt'), denorm: makeFormat(e, m, 'denorm') }),
    [e, m],
  );
  const f = fmts[conv];
  const max = maxFinite(f);
  const xmax = max * 1.06;
  const zoomHi = 2 ** (3 - f.bias);
  const zoom = zoomHi < xmax / 4;
  const pos = values(f).filter((v) => v > 0);
  const intervals = new Set(pos.filter((v) => v >= minNormal(f)).map(binade)).size;
  const gap = pos[0];
  const spacing = pos.length > 1 ? pos[1] - pos[0] : null;

  const setFromSlider = (v: number) => {
    setProbe(v);
    setText(show(v, 4));
  };
  const setFromText = (t: string) => {
    setText(t);
    const v = parseValue(t);
    if (v !== null && Number.isFinite(v) && v >= 0) setProbe(v);
  };

  // Fig. A.4 for the chosen convention: one row per exponent code.
  const rows: { E: number; rule: ReactNode; vals: number[]; spacing: number | null }[] = [];
  for (let E = 0; E < 2 ** e; E++) {
    if (E > topCode(f)) {
      rows.push({ E, rule: 'reserved', vals: [], spacing: null });
      continue;
    }
    const a = anatomy(f, fromFields(f, 0, E, 0));
    const vals = Array.from({ length: 2 ** m }, (_, M) => decode(f, fromFields(f, 0, E, M)).value);
    const rule =
      a.rule === 'flushed-pattern' ? (
        'all read as 0'
      ) : (
        <>
          {a.rule === 'denormal' ? '0.M' : '1.M'} × 2<sup>{minus(a.scale!)}</sup>
        </>
      );
    rows.push({ E, rule, vals, spacing: a.scale === null ? null : 2 ** (a.scale - m) });
  }

  return (
    <Bench
      title="Number line"
      task="Move e and m, switch the convention near 0, and probe a value under all three conventions."
      foot={
        <>
          Book-style formats: 1 sign bit, e exponent bits in excess-(2<sup>e−1</sup> − 1) with the top code reserved, m
          mantissa bits. e = 2, m = 2 is the book's 5-bit format of Fig. A.4. Drawn to scale; dashed boxes are the gap
          between 0 and the smallest positive value, × marks a 0 the format cannot hold, rings mark the probe's nearest value.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Exponent bits e = {e}
          <input type="range" min="1" max="4" value={e} aria-label="exponent bits" onChange={(ev) => setE(+ev.target.value)} />
        </label>
        <label className="field">
          Mantissa bits m = {m}
          <input type="range" min="0" max="4" value={m} aria-label="mantissa bits" onChange={(ev) => setM(+ev.target.value)} />
        </label>
        <label className="field">
          Near 0
          <span className="seg" role="group" aria-label="Convention near 0">
            {CONVS.map((c) => (
              <button key={c.id} aria-pressed={c.id === conv} onClick={() => setConv(c.id)}>
                {c.label}
              </button>
            ))}
          </span>
        </label>
      </div>

      <Lines fmts={fmts} conv={conv} lo={0} hi={xmax} probe={probe} showProbe={revealed} label="Positive representable values under the three conventions" />
      {zoom ? (
        <>
          <div className="mx-label" style={{ margin: '0.6rem 0 0.2rem' }}>
            Near 0, magnified: 0 to {show(zoomHi)}
          </div>
          <Lines fmts={fmts} conv={conv} lo={0} hi={zoomHi} probe={probe} showProbe={revealed} label="The region near 0, magnified" />
        </>
      ) : null}

      <p style={{ margin: '0.7rem 0 0.9rem' }} data-part="summary">
        <b>{CONVS.find((c) => c.id === conv)!.label}:</b> {pos.length} positive values
        {intervals ? <> in {intervals} interval{intervals === 1 ? '' : 's'}</> : null}; smallest {show(gap)}, largest{' '}
        {show(max)}.{' '}
        {conv === 'nozero' ? '0 is not a value. ' : ''}
        {spacing !== null ? (
          <>
            The gap from 0 to the smallest positive value is {show(gap)}, <b data-part="ratio">{show(gap / spacing)}</b> times the
            spacing just above it ({show(spacing)}).
          </>
        ) : null}
      </p>

      <div className="cols">
        <div className="panel">
          <div className="row" style={{ marginBottom: '0.7rem' }}>
            <label className="field">
              Probe x
              <input type="range" min="0" max={xmax} step={xmax / 2000} value={Math.min(probe, xmax)} aria-label="probe slider"
                onChange={(ev) => setFromSlider(+ev.target.value)} />
            </label>
            <label className="field">
              or type it
              <input type="text" inputMode="decimal" value={text} aria-label="probe value" style={{ width: '7rem' }}
                onChange={(ev) => setFromText(ev.target.value)} />
            </label>
          </div>
          {revealed ? (
            <table className="data" style={{ margin: 0 }} aria-label="The probe under the three conventions">
              <thead>
                <tr>
                  <th>Convention</th>
                  <th className="num">Nearest value</th>
                  <th className="num">Rounding error</th>
                </tr>
              </thead>
              <tbody>
                {CONVS.map((c) => {
                  const v = round(fmts[c.id], probe);
                  return (
                    <tr key={c.id} data-conv={c.id} data-nearest={String(v)} data-error={String(Math.abs(v - probe))}
                      style={c.id === conv ? { fontWeight: 650 } : undefined}>
                      <td>{c.label}</td>
                      <td className="num">{show(v)}</td>
                      <td className="num">{show(Math.abs(v - probe), 4)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : (
            <p className="muted" style={{ margin: 0 }} data-part="locked">
              Commit your prediction above to see the probe on the number line and its rounding errors.
            </p>
          )}
        </div>
        <div className="panel">
          <table className="data" style={{ margin: 0 }} aria-label="Values for each exponent code">
            <thead>
              <tr>
                <th>E</th>
                <th>Read as</th>
                <th>Values (M = 0 up)</th>
                <th className="num">Spacing</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.E} data-e={r.E}>
                  <td className="mono">{bitString(r.E, e)}</td>
                  <td>{r.rule}</td>
                  <td>{r.vals.map((v) => show(v, 5)).join(', ')}</td>
                  <td className="num">{r.spacing === null ? '–' : show(r.spacing)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: spacing in real formats
const REAL: Format[] = [FP32, FP16, BF16, E4M3, E5M2, E2M1];
const FP8 = new Set(['e4m3', 'e5m2']);

function statusText(c: Classified): ReactNode {
  switch (c.status) {
    case 'normal':
      return c.exact ? 'normal, exact' : 'normal';
    case 'subnormal':
      return c.exact ? 'subnormal, exact' : 'subnormal';
    case 'zero':
      return c.input === 0 ? 'zero' : <span className="chip fail">underflow to 0</span>;
    case 'flushed':
      return <span className="chip fail">flushed to 0</span>;
    case 'inf':
      return <span className="chip fail">overflow to infinity</span>;
    case 'nan':
      return <span className="chip fail">overflow to NaN</span>;
    case 'saturated':
      return <span className="chip fail">saturated at {show(c.value)}</span>;
  }
}

function RangeChart({ x, ftz }: { x: number; ftz: boolean }) {
  const [ref, W] = useWidth();
  const L0 = W < 560 ? 66 : 92;
  const R0 = W - 14;
  const tlo = -34;
  const thi = 22;
  const T = (t: number) => L0 + ((Math.min(thi, Math.max(tlo, t)) - tlo) / (thi - tlo)) * (R0 - L0);
  const rowH = 24;
  const top = 8;
  const H = top + rowH * REAL.length + 30;
  const tx = x === 0 ? null : Math.log2(Math.abs(x));
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="Range of each format on a log scale" style={{ maxWidth: '100%', height: 'auto', display: 'block' }}>
        {REAL.map((f, r) => {
          const y = top + r * rowH;
          const sub = minSubnormal(f);
          const a = Math.log2(sub ?? minNormal(f));
          const b = Math.log2(minNormal(f));
          const c = Math.log2(maxFinite(f));
          return (
            <g key={f.id} data-format={f.id}>
              <text x={4} y={y + 14} fontSize="11.5">
                {W < 560 ? f.name.replace(/^FP[84] /, '') : f.name}
              </text>
              {sub !== null && !ftz ? (
                <rect x={T(a)} y={y + 5} width={T(b) - T(a)} height={12} style={{ fill: 'none', stroke: 'var(--ink-3)' }} strokeDasharray="3 2" />
              ) : null}
              <rect x={T(b)} y={y + 5} width={T(c) - T(b)} height={12} style={{ fill: 'var(--ink-3)', opacity: 0.35 }} />
              {a < tlo ? (
                <text x={L0 + 3} y={y + 15} fontSize="10" style={{ fill: 'var(--ink)' }}>
                  ◀ {`2^${minus(ftz ? b : a)}`}
                </text>
              ) : null}
              {c > thi ? (
                <text x={R0 - 3} y={y + 15} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink)' }}>
                  {`2^${Math.round(c)}`} ▶
                </text>
              ) : null}
            </g>
          );
        })}
        {[-30, -20, -10, 0, 10, 20].map((t) => (
          <g key={t}>
            <line x1={T(t)} x2={T(t)} y1={top} y2={H - 26} style={{ stroke: 'var(--rule-soft)' }} />
            <text x={T(t)} y={H - 10} fontSize="11" textAnchor="middle">{`2^${minus(t)}`}</text>
          </g>
        ))}
        {tx !== null ? (
          <g>
            <line x1={T(tx)} x2={T(tx)} y1={top - 4} y2={H - 26} style={{ stroke: 'var(--ink)' }} strokeWidth={1.5} />
            <text x={T(tx) + 4} y={H - 28} fontSize="11">
              |x|
            </text>
          </g>
        ) : null}
      </svg>
    </div>
  );
}

function SpacingBench() {
  const [t, setT] = useState(0);
  const [text, setText] = useState('1');
  const [x, setX] = useState(1);
  const [ftz, setFtz] = useState(false);
  const [sat, setSat] = useState(false);
  const setFromSlider = (v: number) => {
    setT(v);
    const xv = 2 ** v;
    setX(xv);
    setText(show(xv, 6));
  };
  const setFromText = (s: string) => {
    setText(s);
    const v = parseValue(s);
    if (v === null || !Number.isFinite(v)) return;
    setX(v);
    if (v !== 0) setT(Math.min(20, Math.max(-30, Math.log2(Math.abs(v)))));
  };
  const presets: [string, number][] = [
    ['70,000', 70000],
    ['1.001', 1.001],
    ['0.1', 0.1],
    ['500', 500],
    ['2^-20', 2 ** -20],
  ];
  return (
    <Bench
      title="Spacing in real formats"
      task="Move x across the range, or pick one of the values. Then turn denormals off."
      foot={
        <>
          Each row rounds x to nearest, ties to even. Bars: solid is the normal range, dashed the subnormal range, on a
          log scale. FP8 and FP4 follow OCP's definitions (appA-facts A3, A4); E2M1 has no infinity or NaN, so it always
          saturates. NVIDIA's and AMD's E4M3 conversions saturate by default, and Metal's saturate for E4M3 but not E5M2
          (A5, B2, C4).
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Slide x (log scale)
          <input type="range" min="-30" max="20" step="0.01" value={t} aria-label="log2 of x" onChange={(ev) => setFromSlider(+ev.target.value)} />
        </label>
        <label className="field">
          or type x
          <input type="text" inputMode="decimal" value={text} aria-label="x" style={{ width: '8rem' }} onChange={(ev) => setFromText(ev.target.value)} />
        </label>
        <span className="row" style={{ gap: '0.35rem' }}>
          {presets.map(([label, v]) => (
            <button key={label} className="quiet" onClick={() => setFromText(label === '2^-20' ? '2^-20' : String(v))}>
              {label}
            </button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="check">
          <input type="checkbox" checked={ftz} onChange={(ev) => setFtz(ev.target.checked)} />
          denormals off (flush to zero)
        </label>
        <label className="check">
          <input type="checkbox" checked={sat} onChange={(ev) => setSat(ev.target.checked)} />
          FP8: saturating conversion
        </label>
      </div>
      <RangeChart x={x} ftz={ftz} />
      <table className="data" style={{ margin: '0.8rem 0 0' }} aria-label="x in each format">
        <thead>
          <tr>
            <th>Format</th>
            <th className="num">Stored value</th>
            <th className="num">Relative error</th>
            <th className="num">Spacing at x (relative)</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {REAL.map((f) => {
            const c = classify(f, x, { flushDenormals: ftz, overflow: sat && FP8.has(f.id) ? 'saturate' : 'ieee' });
            const rel = x === 0 ? NaN : ulp(f, x) / Math.abs(x);
            return (
              <tr key={f.id} data-format={f.id} data-value={String(c.value)} data-status={c.status}>
                <td>{f.name}</td>
                <td className="num" data-part="stored">{show(c.value)}</td>
                <td className="num">{Number.isNaN(c.relError) ? '–' : c.exact ? '0' : show(c.relError, 2)}</td>
                <td className="num">
                  {show(ulp(f, x), 3)}
                  {Number.isNaN(rel) ? null : <span className="muted"> ({show(rel, 2)})</span>}
                </td>
                <td>{statusText(c)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSA2: Record<string, Question> = {
  'appA/A.2/q-ex1': {
    kind: 'numeric',
    id: 'appA/A.2/q-ex1',
    prompt: (
      <>
        A 6-bit format has 1 sign bit, a 2-bit exponent (excess-1, top code reserved) and a 3-bit mantissa, with no zero,
        like Fig. A.5. How many positive values can it represent?
      </>
    ),
    answer: 24,
    tol: 0,
    explain: (
      <p>
        The exponent bits did not change, so there are still three intervals: [0.5, 1), [1, 2) and [2, 4). Each now holds 2
        <sup>3</sup> = 8 values instead of 4, at half the spacing: 24. The range barely moves (the largest value goes from
        3.5 to 3.75), and the gap next to 0 is now 8 times the spacing above it. An extra mantissa bit buys resolution, not
        range.
      </p>
    ),
    source: 'Exercise 1, p. 599; the answer is ours',
  },
  'appA/A.2/q-ex2': {
    kind: 'numeric',
    id: 'appA/A.2/q-ex2',
    prompt: (
      <>
        Now 1 sign bit, a 3-bit exponent (excess-3, top code reserved) and a 2-bit mantissa, no zero. What is the largest
        representable value?
      </>
    ),
    answer: 14,
    tol: 0,
    explain: (
      <p>
        Codes 000 to 110 give exponents −3 to 3: seven intervals starting at 1/8, 1/4, 1/2, 1, 2, 4 and 8. The top one holds
        8, 10, 12 and 14. An extra exponent bit roughly doubles the number of intervals, stretching the range toward both
        larger and smaller magnitudes, and leaves the points per interval alone.
      </p>
    ),
    source: 'Exercise 2, p. 599; the answer is ours',
  },
  'appA/A.2/q-int-half': {
    kind: 'numeric',
    id: 'appA/A.2/q-int-half',
    prompt: <>What is the smallest positive integer that FP16 cannot represent exactly?</>,
    answer: 2049,
    tol: 0,
    explain: (
      <p>
        FP16 has 11 significant bits (10 stored, 1 hidden). Every integer up to 2<sup>11</sup> = 2048 fits. In the interval
        from 2048 to 4096 the spacing is 2, so 2049 falls between 2048 and 2050. BF16's 8 significant bits stop at 256, so it
        first misses 257; FP32's 24 stop at 16,777,216, so it first misses 16,777,217. An index or a count kept in a float
        type goes wrong silently past those points.
      </p>
    ),
    source: 'our arithmetic from the §A.2 observations (mantissa bits set the points per interval)',
  },
  'appA/A.2/q-range-vs-bits': {
    kind: 'choice',
    id: 'appA/A.2/q-range-vs-bits',
    fixedOrder: true,
    prompt: <>Which format keeps both a value of 100,000 and the difference between 1000 and 1001?</>,
    options: [
      { label: 'FP16', why: 'Near 1000 FP16 values are 0.5 apart, so 1001 is exact; but its largest value is 65504, and 100,000 becomes infinity.' },
      { label: 'BF16', why: "BF16 has FP32's range, so 100,000 survives (as 99,840). But near 1000 its values are 4 apart, and 1001 rounds to 1000." },
      { label: 'Both', why: 'Each loses one of the two: FP16 the range, BF16 the resolution.' },
      { label: 'FP32 only', correct: true },
    ],
    explain: (
      <p>
        Between 512 and 1024 a format's values are 2<sup>9</sup> × 2<sup>−m</sup> apart: 2<sup>−1</sup> for FP16 (m =
        10), 2<sup>2</sup> = 4 for BF16 (m = 7). FP16 spends its bits on resolution and stops at 65504; BF16 spends them
        on range, up to about 3.4 × 10<sup>38</sup>. Only FP32 has both. Type each value into the explorer above to
        check.
      </p>
    ),
    source: '§A.2 (range from exponent bits, precision from mantissa bits); NVIDIA Table 51 for the limits (appA-facts A2)',
  },
  'appA/A.2/q-ftz': {
    kind: 'choice',
    id: 'appA/A.2/q-ftz',
    prompt: (
      <>
        What does <code>nvcc --ftz=true</code> do?
      </>
    ),
    options: [
      {
        label: 'Flushes single-precision denormals to zero, which turns denormal support off. The default is false; --use_fast_math implies true.',
        correct: true,
      },
      {
        label: 'Turns denormal support on; without it, denormals are flushed.',
        why: "That is the book's \"(on)\" read literally. NVIDIA's documentation: --ftz=true flushes denormals, and the default, false, preserves them.",
      },
      {
        label: 'Flushes denormals to zero in every precision, double included.',
        why: 'The flag controls single-precision denormals only.',
      },
      {
        label: 'Makes a float atomicAdd on global memory keep its denormals.',
        why: 'That operation always flushes denormals, whatever the flag says.',
      },
    ],
    explain: (
      <p>
        ftz stands for flush to zero. Set to true, it trades the denormal range for speed, and <code>--use_fast_math</code>{' '}
        sets it for you. If a kernel's results change when you add <code>--use_fast_math</code>, tiny values being flushed
        is one of the suspects.
      </p>
    ),
    source: '§A.2, p. 589; nvcc documentation 4.2.7.8 and the CUDA guide (appA-facts A6; ch10 facts A6)',
  },
};

// ---------------------------------------------------------------- the section
export function SA2() {
  return (
    <>
      <p>
        A format's representable numbers are the values it can store exactly; anything else gets rounded to one of them. A
        3-bit unsigned integer represents 0 to 7 and nothing else: not −1, not 9, not 2.5. Floating-point formats also have
        a finite set, but spread very unevenly, and the shape of that spread is what decides where your errors are.
      </p>
      <Source section="A.2" pages="586" figs="Figs. A.2–A.3" />
      <p>
        To see a whole set at once the book uses a 5-bit format: 1 sign bit, a 2-bit exponent in excess-1 and a 2-bit
        mantissa. Exponent codes 00, 01 and 10 stand for −1, 0 and +1; 11 is reserved. With 2 mantissa bits each exponent
        code gives four values. The formats differ only in what the code E = 00 means, and the book lays out three
        conventions side by side.
      </p>
      <Source section="A.2" pages="586–587" figs="Fig. A.4" />

      <h2>Where is the gap?</h2>
      <p>
        Under the plain formula of §A.1, which the book calls the no-zero convention, E = 00 is read like any other code:
        1.M × 2<sup>−1</sup>, so 0.5, 0.625, 0.75 and 0.875. Before you look at the number line, work out what happens to a
        small value.
      </p>
      <Predict id={P_GAP} prompt="In the 5-bit no-zero format, 0.2 is rounded to the nearest representable value. What is the rounding error?">
        {(v) => (
          <p>
            The positive values start at 0.5, and nothing lies between −0.5 and 0.5, so 0.2 becomes 0.5: an error of{' '}
            <b>0.3</b>.{' '}
            {Math.abs(v - 0.3) < 0.005
              ? 'That is what you had.'
              : Math.abs(v - 0.05) < 0.005
                ? '0.05 is the error with denormals, which this format does not have.'
                : Math.abs(v - 0.2) < 0.005
                  ? '0.2 would be the error of rounding to 0, but this format has no 0.'
                  : 'Check which values lie closest to 0.2.'}{' '}
            The table under the number line now shows the other two conventions: abrupt underflow gives 0 (error 0.2) and
            denormals give 0.25 (error 0.05).
          </p>
        )}
      </Predict>
      <NumberLineBench />
      <p>
        The book's own drawing of these values, Fig. A.5, is not to scale near 0: it draws the stretch from 0 to 1 shorter
        than the one from 1 to 2, so 0.5 to 0.875 are not where they belong (the values themselves are clear from Fig.
        A.4). The explorer is to scale, which is why the empty stretch next to 0 looks as large as it is.
      </p>
      <Simplification>
        <p>
          Book-style formats only: the top exponent code is reserved and there is no infinity, so a probe beyond the largest
          value is clamped to it. A probe exactly halfway across the abrupt-underflow gap goes to 0; the book gives no rule
          for that tie.
        </p>
      </Simplification>

      <h2>Five things the number line shows</h2>
      <p>Set the explorer back to e = 2, m = 2 and no zero, then check each of these.</p>
      <ol>
        <li>
          <b>Exponent bits make the intervals.</b> Each exponent code that holds values starts an interval at a power of two:
          three on each side of 0 here. Move e to 3 and there are seven.
        </li>
        <li>
          <b>Mantissa bits fill them.</b> Every interval holds 2<sup>m</sup> values, whatever its width. Move m to 3: eight
          per interval at half the spacing, and the intervals stay where they were. More mantissa bits, smaller rounding
          errors.
        </li>
        <li>
          <b>0 is missing.</b> In the no-zero convention no pattern means 0. The book calls this a serious deficiency.
        </li>
        <li>
          <b>Values crowd toward 0.</b> Each interval is half as wide as the next one out and holds as many values, so the
          largest rounding error in an interval shrinks with the size of the numbers in it. That is the right shape: an error
          of one unit hardly matters on a value of a million and is everything on a value of two.
        </li>
        <li>
          <b>Except right next to 0.</b> The gap between 0 and the smallest positive value is 2<sup>m</sup> times the
          spacing just above it (the explorer's ratio). Values in the gap are rounded with errors 2<sup>m</sup> times larger
          than values just outside it, which hurts code that tests for convergence on tiny values or divides by them.
        </li>
      </ol>
      <Source section="A.2" pages="587–588" figs="Fig. A.5" />
      <Aside>
        <p>
          <b>In the PDF.</b> p. 588 says 0 is missing from "the no-zero column of Fig. A.5". The columns are in Fig. A.4;
          Fig. A.5 is the number line.
        </p>
      </Aside>
      <Ask q={questionsSA2['appA/A.2/q-ex1']} />
      <Ask q={questionsSA2['appA/A.2/q-ex2']} />

      <h2>Abrupt underflow</h2>
      <p>
        The bluntest fix: whenever E is 0, the value is 0, whatever the mantissa says. In the 5-bit format that turns eight
        patterns, four on each side, into 0. Zero is now a value, but the gap gets worse: the smallest positive value is now
        1, so the gap next to 0 is twice as wide as before. Switch the explorer to abrupt underflow and the ratio stays 2
        <sup>m</sup>, measured against a gap that has doubled. The book says some 1980s minicomputers worked this way, and
        some arithmetic units that must be very fast still do.
      </p>
      <Source section="A.2" pages="588–589" figs="Fig. A.6" />

      <h2>Denormals</h2>
      <p>
        IEEE's answer is denormalisation. When E is 0, the mantissa is read as 0.M instead of 1.M, and the exponent is taken
        from the interval above, 0 instead of −1 here. Pattern 0 00 01 then means 0.01B × 2<sup>0</sup> = 0.25. The four
        patterns with E = 00 now spread evenly over [0, 1) at the same spacing as [1, 2): 0, 0.25, 0.5, 0.75. The gap is
        gone, and the explorer's ratio drops to 1.
      </p>
      <Source section="A.2" pages="589" figs="Fig. A.4 (right columns), Fig. A.7" />
      <p>
        For an n-bit exponent field, a pattern with E = 0 means 0.M × 2<sup>−2<sup>n−1</sup> + 2</sup>. NVIDIA writes the same
        thing as 0.M × 2<sup>1 − bias</sup>; with bias = 2<sup>n−1</sup> − 1 the two exponents are equal.
      </p>
      <Aside>
        <p>
          <b>In the PDF.</b> p. 589 points to Fig. A.6 for denormals; Fig. A.7 is meant. Its "previous interval" means the
          next larger one, as the formula confirms. And Fig. A.7 draws a 13th star near 4, which the format cannot hold:
          E = 11 is reserved.
        </p>
      </Aside>

      <h2>What denormals cost, and the switch</h2>
      <p>
        The price is in the hardware: a unit has to notice that a value is in the denormal range and switch representation.
        The book says implementations with moderate hardware can spend thousands of clock cycles each time a denormal is
        produced or consumed, which is why early CUDA devices left them out, and that virtually all recent CUDA devices
        support them.
      </p>
      <Source section="A.2" pages="589" />
      <p>
        It adds that an nvcc compiler flag switches denormals, and writes the setting as "ftz = true (on) or ftz = false
        (off)". Check which way round that is before you rely on it.
      </p>
      <Today checked="2026-11-20 against the nvcc 13.5 and CUDA guide, Clang, MSL 4.1 and PyTorch 2.15 docs (appA-facts A6, B3, C2, D1; ch10 facts A6)">
        <p>
          The book's "ftz = true (on)" reads backwards: <code>--ftz=true</code> turns flushing on and denormals off. NVIDIA
          says only that denormals are "computationally more expensive"; "thousands of clock cycles" is the book's claim.
        </p>
      </Today>
      <p>
        Today every backend lets denormals be lost somewhere, so a kernel whose small values matter should not assume they
        survive:
      </p>
      <table className="data" aria-label="Denormals on each backend">
        <thead>
          <tr>
            <th>Backend</th>
            <th>FP32 denormals</th>
            <th>Exceptions you should know</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>CUDA</td>
            <td>
              Kept by default. <code>--ftz=true</code> flushes them (single precision only); <code>--use_fast_math</code>{' '}
              implies it.
            </td>
            <td>A float <code>atomicAdd</code> on global memory always flushes, whatever the flag.</td>
          </tr>
          <tr>
            <td>HIP</td>
            <td>
              Default not documented. Clang's <code>-fgpu-flush-denormals-to-zero</code> flushes them.
            </td>
            <td>PyTorch documents that MI200's FP16 and BF16 matrix instructions flush denormals.</td>
          </tr>
          <tr>
            <td>Metal</td>
            <td>The specification says denormals "may be flushed to zero", for float, half and bfloat.</td>
            <td>Which Apple GPUs flush is not documented.</td>
          </tr>
        </tbody>
      </table>
      <p>
        Lab A.1 runs on your own GPUs and reports what they do with a value below the smallest normal FP32 number (
        <a href={href('appA', 'labs')}>labs</a>).
      </p>
      <Ask q={questionsSA2['appA/A.2/q-ftz']} />

      <h2>Precision</h2>
      <p>
        The book defines the precision of a format as the largest error it can introduce by storing a number as one of its
        representable values. Smaller error, higher precision. Each extra mantissa bit doubles the values per interval and so
        halves that error; that, and nothing more mysterious, is the difference between single and double precision.
      </p>
      <Source section="A.2" pages="590" />

      <h2>Real formats on one picture</h2>
      <p>
        Everything above carries over to the formats you use. In the normal range the spacing near x is 2<sup>−m</sup> times
        the power of two just below x, so the spacing relative to the value stays within a factor of two of 2<sup>−m</sup>{' '}
        everywhere: 2<sup>−23</sup> for FP32, 2<sup>−10</sup> for FP16, 2<sup>−7</sup> for BF16. Range is where the formats
        differ, and it comes from the exponent bits: FP16 runs from 2<sup>−14</sup> to 65504 in normal values, BF16 and FP32
        from 2<sup>−126</sup> to about 3.4 × 10<sup>38</sup>.
      </p>
      <Beyond>
        <p>
          Real formats are ours, from NVIDIA's Table 51 (its epsilon column is 2<sup>−m</sup>) and OCP's FP8 and MX
          specifications (appA-facts A2 to A4). §A.5 says why the narrow ones exist.
        </p>
      </Beyond>
      <SpacingBench />
      <Ask q={questionsSA2['appA/A.2/q-range-vs-bits']} />
      <p>
        Two more values make the trade concrete. An activation of 70,000 becomes infinity in FP16 and survives in BF16. The
        difference between 1 and 1.001 survives in FP16 and vanishes in BF16. Neither format is more accurate than the other
        in general; they spend 16 bits differently, and FP16 runs out of range long before BF16 runs out of bits.
      </p>
      <Simplification>
        <p>
          Flush-to-zero is modelled as rounding with denormals and then replacing a subnormal result by zero. Hardware may
          also flush inputs, and vendors document different modes. Every conversion here rounds to nearest even.
        </p>
      </Simplification>
      <Ask q={questionsSA2['appA/A.2/q-int-half']} />
    </>
  );
}
