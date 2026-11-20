import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  FILTER_WIDTHS, basicIntensity, ceiling, constIntensity, idealIntensity, multipleOfCeiling, regime,
  smallestComputeBoundWidth, threshold, type Dims, type Hardware,
} from '../../sim/convai';

// Section 7.3: how many operations per byte a convolution could have, how many the
// basic kernel asks for, and why measurements land above the model. Every number on
// the page comes from sim/convai.ts; device numbers come from the reader's reports.

const LAB71 = 'ch07/lab1_constconv';

// ---------------------------------------------------------------- hardware
interface Device extends Hardware {
  key: string;
  label: string;
  note: string;
}
// The book's worked example (ch. 5, p. 94; used again on p. 167). Only ever offered as that.
const BOOK: Device = {
  key: 'book',
  label: "The book's example (Hopper H100)",
  peak: 66900,
  bw: 3350,
  note: "The figures the book quotes in chapter 5 (p. 94): 66.9 TFLOPS single precision, 3.35 TB/s.",
};

function lab51(reports: Reports, b: string): Hardware | null {
  const r = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
  const peak = num(r?.compute_gflops);
  const bw = num(r?.bandwidth_gb_per_s);
  return peak && bw ? { peak, bw } : null;
}

function measuredDevices(reports: Reports): Device[] {
  return realBackends(reports).flatMap((b) => {
    const hw = lab51(reports, b);
    if (!hw) return [];
    return [
      {
        key: b,
        label: `Yours: ${String(reports[b].device.name ?? b)} (${b})`,
        ...hw,
        note: `Measured by lab 5.1 on ${reports[b].generated.slice(0, 10)}: two simple probe kernels, not the vendor's peak figures.`,
      },
    ];
  });
}

const BYTES = [
  { b: 4, label: '4 (float32)' },
  { b: 2, label: '2 (fp16, bf16)' },
  { b: 1, label: '1 (int8)' },
] as const;

const filterName = (m: number, dims: Dims) => (dims === 2 ? `${m}×${m}` : `${m}-tap`);

// ---------------------------------------------------------------- the chart
function IntensityChart({ m, bytes, dims, thr }: { m: number; bytes: number; dims: Dims; thr: number }) {
  const W = 760;
  const H = 330;
  const pad = { l: 62, r: 132, t: 14, b: 48 };
  const yMin = 0.1;
  const yMax = Math.max(400, thr * 2);
  const X = (w: number) => pad.l + ((w - 1) / 14) * (W - pad.l - pad.r);
  const Y = (v: number) =>
    H - pad.b - ((Math.log10(Math.max(v, yMin)) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * (H - pad.t - pad.b);
  const ideal = FILTER_WIDTHS.map((w) => ({ w, v: idealIntensity(w, bytes, dims) }));
  const basic = basicIntensity(bytes);
  const cst = constIntensity(bytes);
  const yt: number[] = [];
  for (let e = Math.ceil(Math.log10(yMin)); e <= Math.floor(Math.log10(yMax)); e++) yt.push(10 ** e);
  const cur = idealIntensity(m, bytes, dims);
  const right = W - pad.r;
  // Labels at the right edge, nudged apart when two lines are close.
  const labels = [
    { key: 'thr', y: Y(thr), text: `threshold ${fmt(thr)}`, color: 'var(--ink)' },
    { key: 'const', y: Y(cst), text: `constant ${fmt(cst)}`, color: 'var(--const)' },
    { key: 'basic', y: Y(basic), text: `basic ${fmt(basic)}`, color: 'var(--global)' },
    { key: 'ideal', y: Y(ideal[ideal.length - 1].v), text: 'ideal', color: 'var(--ink)' },
  ].sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 14) labels[i].y = labels[i - 1].y + 14;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Intensity against filter width" style={{ width: '100%', maxWidth: W, minWidth: 520, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={right - pad.l} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      <rect x={pad.l} y={Y(thr)} width={right - pad.l} height={Math.max(0, H - pad.b - Y(thr))} fill="var(--global-wash)" />
      {yt.map((v) => (
        <g key={'y' + v}>
          <line x1={pad.l} x2={right} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 7} y={Y(v) + 4} fontSize="12" textAnchor="end">{v}</text>
        </g>
      ))}
      {FILTER_WIDTHS.map((w) => (
        <g key={'x' + w}>
          <line x1={X(w)} x2={X(w)} y1={H - pad.b} y2={H - pad.b + 4} stroke="var(--ink-3)" />
          <text x={X(w)} y={H - pad.b + 18} fontSize="12" textAnchor="middle" fontWeight={w === m ? 700 : 400}>{w}</text>
        </g>
      ))}
      <text x={(pad.l + right) / 2} y={H - 8} fontSize="12.5" textAnchor="middle">
        filter width m {dims === 2 ? '(an m × m filter)' : '(m taps)'}
      </text>
      <text x={16} y={(H - pad.b + pad.t) / 2} fontSize="12.5" textAnchor="middle" transform={`rotate(-90 16 ${(H - pad.b + pad.t) / 2})`}>
        FLOP per byte (log scale)
      </text>
      {/* the device's threshold */}
      <line x1={pad.l} x2={right} y1={Y(thr)} y2={Y(thr)} stroke="var(--ink)" strokeDasharray="5 4" strokeWidth="1.4" />
      <text x={pad.l + 8} y={Y(thr) + 16} fontSize="12" style={{ fill: 'var(--global)' }}>memory-bound below the threshold</text>
      <text x={pad.l + 8} y={Y(thr) - 7} fontSize="12" style={{ fill: 'var(--reg)' }}>compute-bound above</text>
      {/* the two kernels as written: flat */}
      <line x1={pad.l} x2={right} y1={Y(basic)} y2={Y(basic)} stroke="var(--global)" strokeWidth="2.2" />
      <line x1={pad.l} x2={right} y1={Y(cst)} y2={Y(cst)} stroke="var(--const)" strokeWidth="2.2" />
      {/* the ideal: grows with the filter */}
      <polyline points={ideal.map((p) => `${X(p.w)},${Y(p.v)}`).join(' ')} fill="none" stroke="var(--ink)" strokeWidth="2.2" />
      {ideal.map((p) => (
        <circle key={p.w} cx={X(p.w)} cy={Y(p.v)} r={p.w === m ? 6.5 : 4} fill={p.w === m ? 'var(--ink)' : 'var(--page)'} stroke="var(--ink)" strokeWidth="1.8" />
      ))}
      {labels.map((l) => (
        <text key={l.key} x={right + 8} y={l.y + 4} fontSize="12" style={{ fill: l.color }} data-label={l.key}
          data-threshold={l.key === 'thr' ? fmt(thr) : undefined}>{l.text}</text>
      ))}
      {/* the gap at the chosen width */}
      <line x1={X(m)} x2={X(m)} y1={Y(cur)} y2={Y(basic)} stroke="var(--ink-2)" strokeDasharray="2 3" />
      <text x={X(m) + (m >= 13 ? -9 : 9)} y={(Y(cur) + Y(basic)) / 2 + 4} fontSize="12.5" fontWeight="600" textAnchor={m >= 13 ? 'end' : 'start'} data-gap>
        gap ×{fmt(cur / basic)}
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------- explorer: ideal against what the kernel asks
function IdealExplorer() {
  const { reports } = useReports();
  const options = useMemo(() => [BOOK, ...measuredDevices(reports)], [reports]);
  const [key, setKey] = useState('book');
  const [custom, setCustom] = useState({ peak: 1000, bw: 100 });
  const [wi, setWi] = useState(2); // index into FILTER_WIDTHS: m = 5
  const [bytes, setBytes] = useState<number>(4);
  const [dims, setDims] = useState<Dims>(2);
  const hw: Device =
    key === 'custom'
      ? { key: 'custom', label: 'Custom', ...custom, note: 'Numbers you typed.' }
      : (options.find((o) => o.key === key) ?? BOOK);
  const m = FILTER_WIDTHS[wi];
  const thr = threshold(hw);
  const rows = [
    { id: 'basic', label: 'Basic kernel (Fig. 7.7)', ai: basicIntensity(bytes) },
    { id: 'const', label: 'Filter in constant memory (Fig. 7.9, next section)', ai: constIntensity(bytes) },
    { id: 'ideal', label: `Ideal for a ${filterName(m, dims)} filter`, ai: idealIntensity(m, bytes, dims) },
  ];
  const first = smallestComputeBoundWidth(thr, bytes, dims);
  return (
    <Bench
      title="Ideal against what the kernel asks"
      task="Find the smallest filter that could ever be compute-bound on the book's example device, then halve the bytes."
      foot={
        <>
          The ideal counts each input and output element once and drops the filter's own bytes; for a 1024-wide image
          the exact figure at this setting is {fmt(idealIntensity(m, bytes, dims, 1024), 6)}. Ceilings are min(peak,
          bandwidth × intensity). {hw.note}
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Filter width m = {m} (radius {(m - 1) / 2})
          <input type="range" min="0" max={FILTER_WIDTHS.length - 1} value={wi} onChange={(e) => setWi(+e.target.value)} aria-label="filter width" />
        </label>
        <label className="field">
          Bytes per element
          <span className="seg" data-control="bytes">
            {BYTES.map((o) => (
              <button key={o.b} aria-pressed={bytes === o.b} onClick={() => setBytes(o.b)}>{o.label}</button>
            ))}
          </span>
        </label>
        <label className="field">
          Dimensions
          <span className="seg" data-control="dims">
            {([2, 1] as Dims[]).map((d) => (
              <button key={d} aria-pressed={dims === d} onClick={() => setDims(d)}>{d === 2 ? '2D image' : '1D signal'}</button>
            ))}
          </span>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Hardware
          <select value={key} onChange={(e) => setKey(e.target.value)} aria-label="hardware">
            {options.map((o) => (
              <option key={o.key} value={o.key}>{o.label}</option>
            ))}
            <option value="custom">Custom numbers</option>
          </select>
        </label>
        {key === 'custom' ? (
          <>
            <label className="field">
              Peak throughput (GFLOP/s)
              <input type="number" min="1" value={custom.peak} onChange={(e) => setCustom({ ...custom, peak: Math.max(1, +e.target.value) })} />
            </label>
            <label className="field">
              Memory bandwidth (GB/s)
              <input type="number" min="1" value={custom.bw} onChange={(e) => setCustom({ ...custom, bw: Math.max(1, +e.target.value) })} />
            </label>
          </>
        ) : null}
      </div>
      <IntensityChart m={m} bytes={bytes} dims={dims} thr={thr} />
      <div className="cols" style={{ marginTop: '0.9rem', gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)' }}>
        <div className="panel" data-panel="ceilings">
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th>Kernel</th>
                <th className="num">FLOP/B</th>
                <th>Regime</th>
                <th className="num">Ceiling, GFLOP/s</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const g = regime(r.ai, thr);
                return (
                  <tr key={r.id} data-row={r.id} data-ai={r.ai} data-regime={g}>
                    <td>{r.label}</td>
                    <td className="num">{fmt(r.ai)}</td>
                    <td>
                      <span className={'chip ' + (g === 'memory-bound' ? 'global' : 'reg')}>{g}</span>
                    </td>
                    <td className="num">{fmt(ceiling(r.ai, hw))}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.7rem 0 0' }}>
            Peak {fmt(hw.peak)} GFLOP/s, bandwidth {fmt(hw.bw)} GB/s, threshold {fmt(thr)} FLOP/B.
          </p>
        </div>
        <div className="panel">
          <p style={{ margin: 0 }} data-first={first}>
            Smallest {dims === 2 ? 'odd filter width' : 'odd number of taps'} whose ideal reaches this threshold at{' '}
            {bytes} byte{bytes > 1 ? 's' : ''} per element: <b>{first}</b>
            {first > FILTER_WIDTHS[FILTER_WIDTHS.length - 1] ? ', beyond this chart' : ''}. Below it, no kernel can be
            compute-bound here.
          </p>
          <p style={{ margin: '0.7rem 0 0' }}>
            The ideal is {fmt(idealIntensity(m, bytes, dims) / basicIntensity(bytes))} times what the basic kernel asks
            for: {dims === 2 ? 'm²' : 'm'}, the number of outputs each input element takes part in.
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- your kernels against their ceilings
interface ConvRow {
  radius: number;
  m: number;
  gflops: number;
  ai: number;
}
function convRows(reports: Reports, b: string, name: 'basic' | 'const'): ConvRow[] {
  const fallback = name === 'basic' ? basicIntensity() : constIntensity();
  return benchRows(reports[b], LAB71)
    .filter((r) => r.name === name)
    .flatMap((r) => {
      const radius = num(r.radius);
      const gflops = num(r.gflops);
      if (radius == null || gflops == null) return [];
      return [{ radius, m: num(r.m) ?? 2 * radius + 1, gflops, ai: num(r.model_intensity) ?? fallback }];
    })
    .sort((a, b2) => a.radius - b2.radius);
}

// Above its model ceiling is not a failure (caches served the reuse), so no colour either way.
function Multiple({ x }: { x: number }) {
  const above = x > 1.02;
  return (
    <span className="chip" data-multiple={x.toFixed(3)}>
      {fmt(x)}× {above ? 'above' : 'of'} its ceiling
    </span>
  );
}

/** Lab 5.1's naive matmul against the ceiling its 0.25 FLOP/B gives it, on each device that has run it. */
function Lab51Naive() {
  const { reports, loading } = useReports();
  const rows = realBackends(reports)
    .filter((b) => b !== 'race')
    .flatMap((b) => {
      const r = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
      const g = num(r?.naive_gflops);
      const c = num(r?.naive_model_ceiling_gflops);
      return g && c ? [{ b, device: String(reports[b].device.name ?? b), x: g / c }] : [];
    });
  if (loading || !rows.length) return <span data-lab51-naive="">lab 5.1 measures the same thing on your device</span>;
  return (
    <span data-lab51-naive={rows.map((r) => r.x.toFixed(3)).join(',')}>
      on your device, lab 5.1&rsquo;s naive kernel ran{' '}
      {rows.map((r) => `${fmt(r.x, 2)} times the ceiling its 0.25 FLOP/B gave it on ${r.device} (${r.b})`).join(', and ')}
    </span>
  );
}

/** Constant over basic, as measured, beside the model's ratio of the two ceilings. */
function SpeedupLine({ basic, cst }: { basic: ConvRow[]; cst: ConvRow[] }) {
  const ratios = basic.flatMap((a) => {
    const c = cst.find((x) => x.radius === a.radius);
    return c ? [c.gflops / a.gflops] : [];
  });
  if (!ratios.length) return null;
  const lo = Math.min(...ratios);
  const hi = Math.max(...ratios);
  return (
    <p style={{ margin: '0 0 0.5rem' }} data-speedup={`${lo.toFixed(3)}-${hi.toFixed(3)}`}>
      The constant-memory kernel ran {fmt(lo)}
      {hi - lo > 0.005 ? <> to {fmt(hi)}</> : null} times as fast as the basic one here; the ratio of their model
      ceilings is 2. A ratio well under 2 suggests the caches were already serving most of the basic kernel&rsquo;s
      filter reads, which left constant memory little to save.
    </p>
  );
}

function YourKernels() {
  const { reports, loading } = useReports();
  const backends = realBackends(reports).filter((b) => convRows(reports, b, 'basic').length || convRows(reports, b, 'const').length);
  return (
    <Bench
      title="Your kernels against their ceilings"
      task="Read only. Where lab 7.1 has run, each measurement is set against the ceiling its model intensity gives it on the same device."
      foot={
        <>
          Ceiling = min(peak, bandwidth × model intensity), with peak and bandwidth from lab 5.1 on the same backend.
          GFLOP/s counts 2·m² operations per output, ghost taps included, as the book does. Lab 7.1's reference
          solution; your own kernels are on the <a href={href('ch07', 'labs')}>labs page</a>.
        </>
      }
    >
      {loading ? (
        <p className="muted" style={{ margin: 0 }}>Reading this machine's lab reports.</p>
      ) : !backends.length ? (
        <p style={{ margin: 0 }} data-empty>
          No measurement yet. Run <code>python3 labs/pmpp.py run 7.1 --solution --bench</code> (and lab 5.1 with{' '}
          <code>--bench</code> if you have not), then reload this page.
        </p>
      ) : (
        backends.map((b) => {
          const hw = lab51(reports, b);
          const basic = convRows(reports, b, 'basic');
          const cst = convRows(reports, b, 'const');
          const radii = [...new Set([...basic, ...cst].map((r) => r.radius))].sort((x, y) => x - y);
          return (
            <div key={b} className="panel" data-backend={b} style={{ marginBottom: '0.8rem' }}>
              <p style={{ margin: '0 0 0.4rem' }}>
                <b>{String(reports[b].device.name ?? b)}</b> ({b})
                {hw ? (
                  <span className="muted"> · lab 5.1: {fmt(hw.bw)} GB/s, {fmt(hw.peak)} GFLOP/s</span>
                ) : (
                  <span className="muted"> · no lab 5.1 bandwidth on this backend: run lab 5.1 with --bench to get the ceilings</span>
                )}
              </p>
              <SpeedupLine basic={basic} cst={cst} />
              <div style={{ overflowX: 'auto' }}>
                <table className="data" style={{ margin: 0 }}>
                  <thead>
                    <tr>
                      <th>Filter</th>
                      <th className="num">Basic, GFLOP/s</th>
                      <th>against 0.25 FLOP/B</th>
                      <th className="num">Constant, GFLOP/s</th>
                      <th>against 0.5 FLOP/B</th>
                    </tr>
                  </thead>
                  <tbody>
                    {radii.map((rad) => {
                      const a = basic.find((x) => x.radius === rad);
                      const c = cst.find((x) => x.radius === rad);
                      return (
                        <tr key={rad} data-radius={rad}>
                          <td>{2 * rad + 1}×{2 * rad + 1}</td>
                          <td className="num">{a ? fmt(a.gflops) : 'n/a'}</td>
                          <td>{a && hw ? <Multiple x={multipleOfCeiling(a.gflops, a.ai, hw)} /> : '·'}</td>
                          <td className="num">{c ? fmt(c.gflops) : 'n/a'}</td>
                          <td>{c && hw ? <Multiple x={multipleOfCeiling(c.gflops, c.ai, hw)} /> : '·'}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          );
        })
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS73: Record<string, Question> = {
  'ch07/7.3/q-regime': {
    kind: 'choice',
    id: 'ch07/7.3/q-regime',
    prompt: (
      <>
        On the book&rsquo;s example device (threshold 20 FLOP/B), take a perfect kernel: every input read once, every
        output written once, 4-byte floats. For which square filters could it be compute-bound?
      </>
    ),
    options: [
      { label: 'Every size, if the kernel is good enough.', why: 'A perfect kernel reaches the ideal and no further. At 3×3 that is 2.25 FLOP/B, a ninth of the threshold.' },
      { label: '7×7 and larger.', why: '7×7 gives 49/4 = 12.25, still below 20.' },
      { label: '9×9 and larger.', correct: true },
      { label: '11×11 and larger.', why: 'The book names 11×11 as compute-bound, but 9×9 already clears the threshold: 81/4 = 20.25.' },
      { label: 'None: convolution is always memory-bound.', why: 'The ideal grows with the filter\'s area, so a large enough filter crosses any fixed threshold.' },
    ],
    explain: (
      <p>
        The ideal is m²/4: 2.25, 6.25 and 12.25 for 3, 5 and 7; 20.25 and 30.25 for 9 and 11. Only from 9×9 does the
        ideal reach 20, and only just. The same numbers are the Ideal column of Fig. 7.14, which is shaded the same
        way. Whether a real kernel gets near its ideal is the subject of section 7.5.
      </p>
    ),
    source: 'ch. 7 §7.3, p. 167; Fig. 7.14, p. 177; notes/today/ch07-facts.md A7',
  },
  'ch07/7.3/q-store': {
    kind: 'choice',
    id: 'ch07/7.3/q-store',
    prompt: <>Why may the book leave the store of <code>P</code> out when it gives the basic kernel 0.25 FLOP/B?</>,
    options: [
      {
        label: 'Each thread stores 4 bytes once but loads 8 bytes per tap, so the store barely moves the ratio.',
        correct: true,
      },
      {
        label: 'Stores write memory rather than read it, and the intensity counts only the bytes a kernel reads.',
        why: 'They are bytes moved like any other. Section 7.5 counts them (p. 176).',
      },
      {
        label: 'The store lands in the L2 cache and stays there until the kernel ends, so it never costs DRAM.',
        why: 'The model counts what the kernel asks of memory and ignores caches for loads and stores alike. Caches are why measurements come out higher, not why the store is left out.',
      },
      {
        label: 'The store overlaps with the next thread\'s loads, so its bytes hide behind traffic already counted.',
        why: 'Overlap can hide time, but the bytes are still moved. Intensity counts bytes.',
      },
    ],
    explain: (
      <p>
        At 5×5 the loop loads 25 × 8 = 200 bytes and the store adds 4: the ratio moves from 0.25 to 50/204 ≈ 0.245, about
        2%. In section 7.5 the loads leave the loop (each thread loads one input element into shared memory), so loads
        and stores become comparable and the store has to be counted.
      </p>
    ),
    source: 'ch. 7 §7.3, p. 167 and §7.5, p. 176',
  },
  'ch07/7.3/q-1d': {
    kind: 'numeric',
    id: 'ch07/7.3/q-1d',
    prompt: <>What is the ideal FLOP/B of a 1D convolution with a 9-tap filter on 4-byte floats? Ignore the filter&rsquo;s own bytes, as the book does in 2D.</>,
    answer: 2.25,
    tol: 0.01,
    unit: 'FLOP/B',
    explain: (
      <p>
        n outputs of 9 taps is 18n operations; reading the input once and writing the output once is 8n bytes. 18n / 8n =
        2.25, which is m/4. In 1D each input element takes part in m outputs, not m², so the ideal grows only linearly
        with the filter: 9 taps in 1D buy what a 3×3 filter buys in 2D.
      </p>
    ),
    source: 'the method of ch. 7 §7.3, p. 167, applied to 1D (our arithmetic)',
  },
  'ch07/7.3/q-fp16': {
    kind: 'numeric',
    id: 'ch07/7.3/q-fp16',
    prompt: <>What is the ideal FLOP/B of a 3×3 convolution whose inputs and outputs are 2-byte values?</>,
    answer: 4.5,
    tol: 0.01,
    unit: 'FLOP/B',
    explain: (
      <p>
        The operations are unchanged, 2·n²·9, and the bytes halve to 2·n²·2, so the ideal is 9/2 = 4.5: m² divided by
        the bytes per element. Every intensity on this page scales the same way, which is why quantised weights help a
        memory-bound kernel even when no arithmetic gets cheaper.
      </p>
    ),
    source: 'the method of ch. 7 §7.3, p. 167; other element sizes are our extension (notes/ch07.md, "Relevance")',
  },
};

// ---------------------------------------------------------------- the page
export function S73() {
  const q = questionsS73;
  return (
    <>
      <p>
        The kernel of section 7.2 is correct. This section asks how fast it can be, with the counting chapter 5 used
        for matrix multiplication: floating-point operations performed per byte of global-memory traffic. The count is
        short, and it gives two numbers that differ by a factor you can name before you run anything.
      </p>
      <Source section="7.3" pages="167" />

      <h2>What the computation needs</h2>
      <p>
        Take an n × n input, an m × m filter with m = 2r + 1, and 4-byte floats. Every output is m² taps of one multiply
        and one add, so the whole convolution is 2·n²·m² operations. Taps that land on ghost cells are counted too, as
        the book counts them.
      </p>
      <p>
        The least traffic any kernel could have reads every input element once, reads the filter once and writes every
        output once: 4n² + 4m² + 4n² bytes. The filter is tiny next to the image, so drop its 4m², and the best
        possible ratio is
      </p>
      <p style={{ textAlign: 'center', fontSize: '1.15rem' }}>
        2·n²·m² / 8n² = <b>m²/4 FLOP/B</b>
      </p>
      <p>
        It grows with the filter&rsquo;s area: 2.25 for a 3 × 3 filter, 30.25 for 11 × 11. The reason is reuse. Each
        input element takes part in m² outputs, and a kernel that fetched it once would get m² uses out of every fetch.
      </p>
      <Source section="7.3" pages="167" />

      <h2>The filter size decides the regime</h2>
      <p>
        Now put those numbers against a device. The book uses the GPU of chapter 5 as its example, whose threshold is
        about 20 FLOP/B. A 3 × 3 convolution reaches only 2.25 even in the ideal case, so no kernel, however well
        written, keeps that device&rsquo;s arithmetic units busy. An 11 × 11 convolution, at 30.25, could reach the
        arithmetic peak, but only if the kernel actually gets close to its ideal, which is what the rest of the
        chapter works on.
      </p>
      <Today checked="2026-11-20 against nvidia.com (notes/today/ch07-facts.md A7, ch05-facts.md A8)">
        <p>
          The book&rsquo;s device is the NVIDIA H100 SXM. With NVIDIA&rsquo;s printed 67 TFLOPS (FP32) and 3.35 TB/s the
          threshold is 20.0 FLOP/B; the book&rsquo;s 66.9 gives 19.97. The memory-bound and compute-bound shading of
          Fig. 7.14 matches it cell for cell. Your own threshold comes from lab 5.1.
        </p>
      </Today>
      <IdealExplorer />
      <p>
        Three things to find in it. Move m and the ideal climbs with the filter&rsquo;s area, while both kernels as
        written stay flat: the gap between them is m². Switch to 2 bytes per element and every line doubles, ideal and
        kernels alike. Switch to 1D and the ideal grows only with m, so a 1D filter in float32 needs 81 taps before the
        book&rsquo;s example device could be compute-bound.
      </p>
      <Simplification>
        <p>
          One peak and one bandwidth per device. Changing the bytes per element changes the traffic only: the peak
          stays the one you picked, although real devices have different peaks for 16-bit and 8-bit arithmetic.
        </p>
        <p>No caches: the lines are what a kernel asks of memory (see below).</p>
      </Simplification>
      <Ask q={q['ch07/7.3/q-regime']} />

      <h2>What the basic kernel asks for</h2>
      <p>
        Now the kernel you have. Its inner statement, in the corrected form of lab 7.1, is{' '}
        <code>Pvalue += F[fRow*(2*r+1) + fCol] * N[inRow*width + inCol];</code>
      </p>
      <Predict
        id="ch07/7.3/p-basic"
        prompt="For an 11 × 11 filter, how many operations does that statement perform per byte it loads from global memory, counting only the loop?"
        unit="FLOP/B"
      >
        {(v) => (
          <p>
            Each iteration loads one element of F and one of N, 8 bytes, for one multiply and one add: 2/8 ={' '}
            <b>0.25 FLOP/B</b>.{' '}
            {Math.abs(v - 0.25) < 0.02
              ? 'Exactly.'
              : v > 1
                ? 'If you had more, you were probably thinking of the ideal: the kernel does not share anything between threads, so it gets none of the reuse.'
                : 'Count bytes, not accesses: each of the two loads moves 4 of them.'}{' '}
            It is the same for 3 × 3: each tap brings its own two loads, so the filter size cancels. Against the ideal of
            30.25, this kernel asks memory for 121 times the bytes it needs.
          </p>
        )}
      </Predict>
      <Source section="7.3" pages="167" />
      <p>
        That factor of m² is exactly the reuse the ideal counts on and the kernel throws away: every thread fetches its
        whole patch again, although its neighbours fetched most of it a moment ago.
      </p>
      <p>
        What about the store? Each thread writes <code>P</code> once, after its loops. The book leaves it out of the
        0.25, and that is fair while the loop moves 8 bytes per tap: at 5 × 5 the store adds 4 bytes to 200. Section 7.5
        has to count stores, because there the loads leave the loop and stop dwarfing them.
      </p>
      <Source section="7.3, 7.5" pages="167, 176" />
      <Ask q={q['ch07/7.3/q-store']} />

      <h2>Measured intensity is higher than this</h2>
      <p>
        The 0.25 charges every load to DRAM. The book says in the same breath that the real figure is higher: a filter
        element is read by every thread and an input element by up to m² threads, so many of those loads are found in
        the L1 or L2 cache instead. Chapter 5 showed the same thing for matrix multiplication: <Lab51Naive />.
      </p>
      <Source section="7.3" pages="168" />
      <p>
        So read the model as a statement about what the kernel asks of memory. What actually reaches DRAM is up to the
        device and its caches, and only a measurement shows it. Lab 7.1 times the basic kernel and the constant-memory
        kernel of the next section at several radii. With lab 5.1&rsquo;s bandwidth for the same device, the panel below
        sets each measurement against its model ceiling.
      </p>
      <YourKernels />

      <h2>The book&rsquo;s answer: on-chip memory, on purpose</h2>
      <p>
        Caches help, but you do not decide what they keep. The book&rsquo;s plan for the rest of the chapter is to put the
        reused data in on-chip memory deliberately: the filter in constant memory, where the constant cache serves it
        (<a href={href('ch07', '7.4')}>section 7.4</a>, which takes the kernel from 0.25 to 0.5), and tiles of the input
        in shared memory (<a href={href('ch07', '7.5')}>section 7.5</a>, which goes after the factor of m²).
      </p>
      <Source section="7.3" pages="168" />

      <h2>Bytes per element, and the 1D case</h2>
      <p>
        Two variations you will meet in inference code. With fewer bytes per element every intensity on this page is
        divided by the bytes, so 16-bit values double all of them and 8-bit values quadruple them, ideal and kernels
        alike. With one dimension instead of two, each input takes part in m outputs instead of m², so the ideal is
        m/4 in float32: it grows with the filter&rsquo;s width, not its area.
      </p>
      <Beyond>
        <p>
          Both are our arithmetic with the book&rsquo;s method; the book fixes 2D and 4-byte floats. A 4-tap 1D filter in
          float32 has an ideal of 1 FLOP/B, far below the book&rsquo;s threshold of 20: a kernel like that is judged by the
          bandwidth it reaches. Section 7.5 shows where such a filter runs in LLM serving.
        </p>
      </Beyond>
      <Ask q={q['ch07/7.3/q-1d']} />
      <Ask q={q['ch07/7.3/q-fp16']} />
    </>
  );
}
