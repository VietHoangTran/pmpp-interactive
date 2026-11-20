import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';

// ---------------------------------------------------------------- hardware
interface Hardware {
  key: string;
  label: string;
  /** Peak arithmetic throughput, GFLOP/s. */
  peak: number;
  /** Peak global-memory bandwidth, GB/s. */
  bw: number;
  note: string;
}
// The book's worked example (p. 94). Used only as the book's example.
const BOOK: Hardware = {
  key: 'book',
  label: "The book's example (Hopper H100)",
  peak: 66900,
  bw: 3350,
  note: "Figures the book quotes on p. 94: 66.9 TFLOPS single precision, 3.35 TB/s.",
};

function measured(reports: Reports): Hardware[] {
  return realBackends(reports).flatMap((b) => {
    const r = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
    const peak = num(r?.compute_gflops);
    const bw = num(r?.bandwidth_gb_per_s);
    if (!peak || !bw) return [];
    return [
      {
        key: b,
        label: `Yours: ${String(reports[b].device.name ?? b)} (${b})`,
        peak,
        bw,
        note: `Measured by lab 5.1 on ${reports[b].generated.slice(0, 10)}: two simple probe kernels, not the vendor's peak figures.`,
      },
    ];
  });
}

// ---------------------------------------------------------------- roofline plot
interface Mark {
  label: string;
  x: number; // FLOP/B
  y?: number; // achieved GFLOP/s, if measured
  tone: 'model' | 'measured';
}

function Roofline({ hw, marks }: { hw: Hardware; marks: Mark[] }) {
  const W = 700;
  const H = 380;
  const pad = { l: 64, r: 16, t: 14, b: 44 };
  const xMin = 0.03;
  const xMax = 3000;
  const yMax = hw.peak * 3;
  const yMin = hw.bw * xMin * 0.6;
  const X = (v: number) => pad.l + ((Math.log10(v) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin))) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * (H - pad.t - pad.b);
  const knee = hw.peak / hw.bw;
  const roof = (x: number) => Math.min(hw.peak, hw.bw * x);
  const xt = [0.1, 1, 10, 100, 1000];
  const yt: number[] = [];
  for (let e = Math.ceil(Math.log10(yMin)); e <= Math.floor(Math.log10(yMax)); e++) yt.push(10 ** e);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Roofline plot" style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {/* the two regimes */}
      <rect x={pad.l} y={pad.t} width={X(knee) - pad.l} height={H - pad.t - pad.b} fill="var(--global-wash)" />
      <text x={pad.l + 8} y={pad.t + 16} fontSize="11" style={{ fill: 'var(--global)' }}>memory-bound</text>
      <text x={X(knee) + 8} y={pad.t + 16} fontSize="11" style={{ fill: 'var(--reg)' }}>compute-bound</text>
      {xt.map((v) => (
        <g key={'x' + v}>
          <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(v)} y={H - pad.b + 15} fontSize="11" textAnchor="middle">{v}</text>
        </g>
      ))}
      {yt.map((v) => (
        <g key={'y' + v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="11" textAnchor="end">{v >= 1000 ? v / 1000 + 'k' : v}</text>
        </g>
      ))}
      <text x={(W + pad.l) / 2} y={H - 8} fontSize="11.5" textAnchor="middle">operations per byte of global-memory traffic (FLOP/B)</text>
      <text x={14} y={(H - pad.b + pad.t) / 2} fontSize="11.5" textAnchor="middle" transform={`rotate(-90 14 ${(H - pad.b + pad.t) / 2})`}>throughput (GFLOP/s)</text>
      {/* the roof */}
      <polyline
        points={`${X(xMin)},${Y(hw.bw * xMin)} ${X(knee)},${Y(hw.peak)} ${X(xMax)},${Y(hw.peak)}`}
        fill="none" stroke="var(--ink)" strokeWidth="2.2" />
      <line x1={X(knee)} x2={X(knee)} y1={Y(hw.peak)} y2={H - pad.b} stroke="var(--ink)" strokeDasharray="3 3" />
      <text x={X(knee)} y={H - pad.b - 6} fontSize="11" textAnchor="middle">threshold {fmt(knee)}</text>
      {marks.map((m, i) => {
        const cx = X(Math.min(Math.max(m.x, xMin), xMax));
        const top = roof(m.x);
        if (m.tone === 'model')
          return (
            <g key={i}>
              <line x1={cx} x2={cx} y1={Y(top)} y2={H - pad.b} stroke="var(--ink-3)" strokeWidth="1" />
              <circle cx={cx} cy={Y(top)} r="4.5" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.8" />
              <text x={cx + 7} y={Y(top) + (i % 2 ? 16 : -8)} fontSize="11">{m.label}</text>
            </g>
          );
        const above = (m.y ?? 0) > top * 1.02;
        return (
          <g key={i}>
            <rect x={cx - 4.5} y={Y(m.y!) - 4.5} width="9" height="9" transform={`rotate(45 ${cx} ${Y(m.y!)})`}
              fill={above ? 'var(--fail)' : 'var(--ok)'} />
            <text x={cx + 9} y={Y(m.y!) + 4} fontSize="11" style={{ fill: above ? 'var(--fail)' : 'var(--ok)' }}>{m.label}</text>
          </g>
        );
      })}
    </svg>
  );
}

function RooflineBench() {
  const { reports } = useReports();
  const options = useMemo(() => [BOOK, ...measured(reports)], [reports]);
  const [key, setKey] = useState('book');
  const [custom, setCustom] = useState({ peak: 1000, bw: 100 });
  const [tile, setTile] = useState(16);
  const [n, setN] = useState(1024);
  const [showMeasured, setShowMeasured] = useState(true);
  const hw: Hardware =
    key === 'custom'
      ? { key: 'custom', label: 'Custom', ...custom, note: 'Numbers you typed.' }
      : (options.find((o) => o.key === key) ?? BOOK);
  const knee = hw.peak / hw.bw;

  const marks: Mark[] = [
    { label: 'vector add', x: 1 / 12, tone: 'model' },
    { label: 'naive matmul', x: 0.25, tone: 'model' },
    { label: `tiled, T=${tile}`, x: 0.25 * tile, tone: 'model' },
    { label: `ideal matmul, N=${n}`, x: n / 6, tone: 'model' },
  ];
  // Your own measurements, when this hardware is a device lab 5.1 ran on.
  const run = reports[hw.key];
  const naive = num(benchRows(run, 'ch05/lab1_roofline').find((x) => x.name === 'roofline')?.naive_gflops);
  const tiled = benchRows(run, 'ch05/lab2_tiled').filter((x) => x.name === 'tiled');
  if (showMeasured && naive) marks.push({ label: `naive, measured ${fmt(naive)}`, x: 0.25, y: naive, tone: 'measured' });
  if (showMeasured)
    for (const t of tiled) {
      const g = num(t.gflops);
      const T = num(t.tile);
      if (g && T === tile) marks.push({ label: `T=${T}, measured ${fmt(g)}`, x: 0.25 * T, y: g, tone: 'measured' });
    }
  const row = (label: string, x: number) => (
    <tr key={label}>
      <td>{label}</td>
      <td className="num">{fmt(x)}</td>
      <td>{x < knee ? <span className="chip global">memory-bound</span> : <span className="chip reg">compute-bound</span>}</td>
      <td className="num">{fmt(Math.min(hw.peak, hw.bw * x))}</td>
      <td className="num">{fmt((100 * Math.min(hw.peak, hw.bw * x)) / hw.peak)}%</td>
    </tr>
  );
  return (
    <Bench
      title="Roofline"
      task="Pick the hardware, then move the tile size until the tiled kernel crosses the threshold."
      foot={
        <>
          Log scales on both axes. Open circles are ceilings from the model (the best a kernel with that ratio could
          do); filled diamonds are measurements from your labs. {hw.note}
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Hardware
          <select value={key} onChange={(e) => setKey(e.target.value)}>
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
        <label className="field">
          Tile width T = {tile}
          <input type="range" min="1" max="128" value={tile} onChange={(e) => setTile(+e.target.value)} />
        </label>
        <label className="field">
          Matrix width N = {n}
          <input type="range" min="64" max="8192" step="64" value={n} onChange={(e) => setN(+e.target.value)} />
        </label>
        {run ? (
          <label className="check">
            <input type="checkbox" checked={showMeasured} onChange={(e) => setShowMeasured(e.target.checked)} />
            show my measurements
          </label>
        ) : null}
      </div>
      <div className="cols plot">
        <Roofline hw={hw} marks={marks} />
        <div className="panel">
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th>Kernel</th>
                <th className="num">FLOP/B</th>
                <th>Regime</th>
                <th className="num">Ceiling, GFLOP/s</th>
                <th className="num">of peak</th>
              </tr>
            </thead>
            <tbody>{marks.filter((m) => m.tone === 'model').map((m) => row(m.label, m.x))}</tbody>
          </table>
          <p className="muted" style={{ margin: '0.7rem 0 0' }}>
            Peak {fmt(hw.peak)} GFLOP/s, bandwidth {fmt(hw.bw)} GB/s, threshold {fmt(knee)} FLOP/B.
          </p>
          {marks.some((m) => m.tone === 'measured' && (m.y ?? 0) > Math.min(hw.peak, hw.bw * m.x) * 1.02) ? (
            <p style={{ margin: '0.7rem 0 0' }}>
              <span className="chip fail">above the roof</span> A measured point sits above the line the model says
              nothing can cross. The model is not wrong about the hardware; it is wrong about the kernel. See below.
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- linear layer
function LinearLayerBench() {
  const { reports } = useReports();
  const [batch, setBatch] = useState(1);
  const [bytes, setBytes] = useState(4);
  const k = 1024;
  const l = 4096;
  const ai = (j: number, b: number) => (2 * j * k * l) / (k * l * b + j * (k + l) * b);
  const backends = realBackends(reports);
  const rows = backends.flatMap((b) =>
    benchRows(reports[b], 'ch05/lab3_gemm').filter((r) => r.name === 'linear_layer').map((r) => ({ b, r })),
  );
  return (
    <Bench
      title="A linear layer, by batch size"
      task="Find the batch size at which this layer stops being limited by memory on the book's example GPU (threshold 20 FLOP/B)."
      foot="Ideal traffic only: every weight, input and output touched exactly once. Layer shape from the book's own example in §20.6 (p. 505), which uses 16-bit values."
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Batch size (rows through the layer at once) = {batch}
          <input type="range" min="0" max="10" value={Math.log2(batch)} onChange={(e) => setBatch(2 ** +e.target.value)} />
        </label>
        <label className="field">
          Bytes per value
          <span className="seg">
            {[4, 2, 1, 0.5].map((b) => (
              <button key={b} aria-pressed={bytes === b} onClick={() => setBytes(b)}>
                {b === 4 ? '4 (float32)' : b === 2 ? '2 (16-bit)' : b === 1 ? '1 (int8)' : '0.5 (4-bit)'}
              </button>
            ))}
          </span>
        </label>
      </div>
      <div className="cols">
        <div className="panel">
          <p style={{ margin: 0 }}>
            {k} inputs → {l} outputs. Operations: 2 × {batch} × {k} × {l} = {fmt(2 * batch * k * l, 4)}. Bytes:{' '}
            {fmt(k * l * bytes, 4)} of weights + {fmt(batch * (k + l) * bytes, 4)} of activations.
          </p>
          <p style={{ fontSize: '1.5rem', margin: '0.6rem 0 0.2rem', fontWeight: 600 }}>{fmt(ai(batch, bytes))} FLOP/B</p>
          <p className="muted" style={{ margin: 0 }}>
            {ai(batch, bytes) < 20 ? (
              <span className="chip global">memory-bound</span>
            ) : (
              <span className="chip reg">compute-bound</span>
            )}{' '}
            against the book's 20 FLOP/B threshold.
          </p>
        </div>
        <div className="panel">
          {rows.length ? (
            <>
              <p style={{ margin: '0 0 0.4rem' }}>
                <b>Your lab 5.3 measurements</b> (float32, tiled kernel)
              </p>
              <table className="data" style={{ margin: 0 }}>
                <thead>
                  <tr>
                    <th>Backend</th>
                    <th className="num">Batch</th>
                    <th className="num">GFLOP/s</th>
                    <th className="num">Rows per second</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map(({ b, r }, i) => (
                    <tr key={i} style={num(r.batch) === batch ? { background: 'var(--rule-soft)' } : undefined}>
                      <td>{b}</td>
                      <td className="num">{String(r.batch)}</td>
                      <td className="num">{fmt(num(r.gflops) ?? NaN)}</td>
                      <td className="num">{fmt(num(r.rows_per_s) ?? NaN)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          ) : (
            <p style={{ margin: 0 }}>
              No measurements yet. Lab 5.3 times exactly this layer at batch sizes 1 to 1024 on your device (
              <a href={href('ch05', 'labs')}>labs</a>).
            </p>
          )}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questions51: Record<string, Question> = {
  'ch05/5.1/q-regime': {
    kind: 'choice',
    id: 'ch05/5.1/q-regime',
    prompt: (
      <>
        A kernel performs 36 floating-point operations and seven 4-byte global-memory accesses per thread. On device A
        the peaks are 200 GFLOP/s and 100 GB/s; on device B, 300 GFLOP/s and 250 GB/s. Where is it memory-bound?
      </>
    ),
    options: [
      { label: 'On both.', why: 'Compute each threshold: peak throughput divided by peak bandwidth.' },
      { label: 'On A only.', correct: true },
      { label: 'On B only.', why: 'B has more bandwidth per unit of compute, so its threshold is lower, not higher.' },
      { label: 'On neither.', why: 'A needs 2 FLOP/B to keep its cores busy; this kernel supplies 1.29.' },
    ],
    explain: (
      <p>
        The kernel's ratio is 36 / 28 = 1.29 FLOP/B. A's threshold is 200 / 100 = 2.0, so on A the memory system cannot
        feed the cores: memory-bound. B's threshold is 300 / 250 = 1.2, so on B the same kernel is compute-bound. The
        regime belongs to the pair (kernel, device), never to the kernel alone.
      </p>
    ),
    source: 'ch. 5 exercise 9, p. 119',
  },
  'ch05/5.1/q-speedoflight': {
    kind: 'numeric',
    id: 'ch05/5.1/q-speedoflight',
    prompt: (
      <>
        A vector-addition kernel adds two vectors of 250 million floats in 2.0 ms on a device whose global-memory
        bandwidth peaks at 2000 GB/s. What percentage of the speed of light is that?
      </>
    ),
    answer: 75,
    tol: 0.02,
    unit: '%',
    explain: (
      <p>
        Three arrays of 250 million 4-byte floats is 3 GB moved. 3 GB / 2.0 ms = 1500 GB/s, and 1500 / 2000 = 75%. So
        at most a 1.33× speed-up is left for this kernel on this device, however clever the code. That bound is what
        the analysis is for: it tells you when to stop.
      </p>
    ),
    source: 'ch. 5 §5.1, p. 96',
  },
  'ch05/5.1/q-decode': {
    kind: 'choice',
    id: 'ch05/5.1/q-decode',
    prompt: (
      <>
        Generating one token at batch size 1 pushes a single row through each weight matrix (a matrix-vector product).
        Suppose you write the best possible kernel for it. Which statement holds?
      </>
    ),
    options: [
      {
        label: 'Tiling with a large enough tile makes it compute-bound, as it does for square matmul.',
        why: 'Tiling removes repeated fetches of the same element. Here each weight is needed exactly once per token; there is nothing to reuse.',
      },
      {
        label: 'It stays memory-bound: each weight is used once, so the ratio cannot rise above about 2 operations per weight fetched.',
        correct: true,
      },
      {
        label: 'It is compute-bound, because the matrix is large and large problems have high intensity.',
        why: 'For square matmul the ideal ratio grows with N because each element takes part in N products. In a matrix-vector product each weight takes part in one.',
      },
    ],
    explain: (
      <>
        <p>
          One multiply and one add per weight, and at least one fetch of that weight: 2 FLOP per 4 bytes = 0.5 FLOP/B
          for float32, however the kernel is written. The only ways up are to make each weight serve more rows at once
          (batching) or to make each weight smaller (quantisation). Both show up directly in the explorer above.
        </p>
      </>
    ),
    source: 'our analysis, using the §5.1 method; the book reaches the same conclusion for LLM decoding in §20.6, p. 505',
  },
};

export function S51() {
  return (
    <>
      <p>
        A GPU has two speed limits that matter here. One is how many arithmetic operations it can do per second. The
        other is how many bytes it can move in or out of a given memory per second. A kernel runs into whichever one it
        reaches first.
      </p>
      <Source section="5.1" pages="93–94" />
      <p>
        If the cores always have operands waiting, the kernel is <b>compute-bound</b>: it goes as fast as the arithmetic
        units allow. If the cores sit idle waiting for operands to arrive, it is <b>memory-bound</b>: it goes as fast
        as memory can deliver. Which one applies depends on a single property of the kernel: how many floating-point
        operations it performs for each byte it pulls from global memory. The book calls this the
        compute-to-global-memory-access ratio and measures it in FLOP/B; you will also see it called arithmetic
        intensity.
      </p>

      <h2>Count it for a kernel you know</h2>
      <p>
        Vector addition, <code>C[i] = A[i] + B[i]</code>, on 4-byte floats: each thread loads two values, adds them and
        stores one.
      </p>
      <Predict id="ch05/5.1/p-vecadd" prompt="How many floating-point operations does it perform per byte of global memory it touches?" unit="FLOP/B">
        {(v) => (
          <p>
            One addition for 12 bytes (4 loaded, 4 loaded, 4 stored): 1/12 ≈ <b>0.083 FLOP/B</b>.{' '}
            {Math.abs(v - 1 / 12) < 0.01
              ? 'That is what you had.'
              : v > 0.2
                ? 'If you were higher, check whether you counted the store: writing C[i] is traffic too.'
                : 'Count bytes, not accesses: each access moves 4 of them.'}
          </p>
        )}
      </Predict>
      <Source section="5.1" pages="96" />
      <p>
        Now the device side. Divide peak throughput by peak bandwidth and you get the ratio at which the two limits
        meet. A kernel above that <b>threshold</b> gives the cores more work per byte than memory can starve them of; a
        kernel below it leaves cores waiting. For the GPU the book uses as its running example, the threshold is about
        20 FLOP/B. Vector addition at 0.083 is two hundred times short of it.
      </p>
      <Today checked="2026-11-20 against nvidia.com and the CUDA 13.5 docs (notes/today/ch05-facts.md, A8)">
        <p>
          The book's device is an NVIDIA H100. NVIDIA's product page confirms 3.35 TB/s for the SXM variant and prints
          FP32 throughput as 67 teraFLOPS; the book's 66.9 appears in no NVIDIA source we found. Your device has its own
          threshold: lab 5.1 measures it.
        </p>
      </Today>

      <h2>The roofline</h2>
      <p>
        Put the kernel's ratio on the horizontal axis and throughput on the vertical. The hardware draws a roof: a
        slope where bandwidth is the limit (throughput = bandwidth × ratio) and a flat top where arithmetic is. Every
        kernel lives under it. How far under tells you how well it is written; which part of the roof is overhead tells
        you what kind of work would help.
      </p>
      <Source section="5.1" pages="95–96" figs="the Roofline Model sidebar" />
      <RooflineBench />
      <p>
        Three things to notice. A memory-bound kernel sitting near the slope is already efficient, and the only way to
        make it faster is to raise its ratio, moving it right. A kernel far below the roof has room to use the hardware
        better as it is. And the peaks are limits, not promises: nothing guarantees a compute-bound kernel reaches the
        flat part.
      </p>
      <Simplification>
        <p>
          The roof has two straight segments because it has two numbers. Real devices have several memories with
          different bandwidths, and different peaks for different data types. The book's version, like this one, uses
          single precision and global memory only.
        </p>
        <p>The book's sketch uses linear axes; log axes are used here so that 0.08 and 170 fit in one picture.</p>
      </Simplification>
      <Ask q={questions51['ch05/5.1/q-regime']} />

      <h2>Speed of light</h2>
      <p>
        For a memory-bound kernel there is a direct check on how good it is. Add up the bytes it must move, divide by
        the time it took, and compare with the peak bandwidth. The book's example: adding two vectors of a billion
        floats moves 12 GB; finishing in 4 ms means 3 TB/s, which is 90% of the example device's 3.35 TB/s. The kernel
        runs at 90% of the speed of light, and no rewrite can gain more than the remaining 10%.
      </p>
      <Source section="5.1" pages="96" />
      <Ask q={questions51['ch05/5.1/q-speedoflight']} />

      <h2>Matrix multiplication: the gap between what is possible and what you wrote</h2>
      <p>
        Count the work in an N×N multiplication without looking at any code. Each of the N² outputs is a dot product: N
        multiplies and N adds. That is 2N³ operations. If every input element were fetched exactly once and every
        output stored once, the traffic would be 3N² floats, or 12N² bytes. The best possible ratio is therefore
        2N³ / 12N² = N/6 FLOP/B. At N = 1024 that is about 170: far to the right of any threshold. Matrix
        multiplication is, by nature, a computation the arithmetic units could be kept busy with.
      </p>
      <Source section="5.1" pages="96–97" />
      <Predict
        id="ch05/5.1/p-naive"
        prompt={
          <>
            Now the kernel you actually have. Its inner loop is{' '}
            <code>Pvalue += M[row*Width+k] * N[k*Width+col]</code>. What is its ratio, counting only the loop?
          </>
        }
        unit="FLOP/B"
      >
        {(v) => (
          <p>
            Each iteration loads one element of M and one of N (8 bytes) for one multiply and one add:{' '}
            <b>0.25 FLOP/B</b>, whatever N is.{' '}
            {Math.abs(v - 0.25) < 0.02 ? 'Exactly.' : 'The matrix size cancels: more elements, proportionally more fetches.'}{' '}
            The ideal ratio grows with N and this one does not, because the kernel fetches each input element again for
            every thread that needs it. On the book's example device that caps it at 3350 × 0.25 ≈ 840 GFLOP/s, a little
            over 1% of the peak.
          </p>
        )}
      </Predict>
      <p>
        Go back to the roofline, select the book's hardware, and look at the distance between "naive matmul" and "ideal
        matmul". That distance is the subject of this chapter and of chapter 15. The rest of this chapter closes part
        of it by a factor you choose: the tile width.
      </p>

      <h3>When a measurement lands above the roof</h3>
      <p>
        If you have run lab 5.1, select your own device in the explorer. On the machine this course was built on, the
        naive kernel measured well <em>above</em> the ceiling the model gives it; your explorer shows by how much on
        yours. Nothing broke. The 0.25
        figure charges every load to global memory, and the book says in the same breath that the true ratio is higher
        because some of those loads are served by on-chip caches. How much higher depends on the device and the matrix
        size, and it can be a lot.
      </p>
      <Source section="5.1" pages="97" />
      <p>
        Hold on to both halves of that. The model tells you what a kernel <em>asks</em> of memory. Only a measurement
        tells you what the device actually had to fetch. When this chapter says tiling cuts traffic by a factor of T,
        that is a statement about the asking; the speed-up you measure in lab 5.2 will be smaller wherever caches were
        already doing part of the job.
      </p>

      <h2>Underneath a linear layer</h2>
      <p>
        The same counting explains two things you already know from the framework side: why generation at batch size 1
        is slow per unit of compute, and why quantisation speeds it up even when no arithmetic gets cheaper.
      </p>
      <Beyond>
        <p>
          This applies the book's §5.1 method to a rectangular product. The book does it itself much later, for LLM
          inference (§20.6, p. 505), with the same layer shape used here.
        </p>
      </Beyond>
      <LinearLayerBench />
      <p>
        At batch size 1 with 16-bit values the explorer gives 1.0 FLOP/B, the figure the book quotes; at batch 512 it
        gives 315, also the book's. Weights dominate the traffic, and a weight fetched once can serve every row of the
        batch. Halving the bytes per weight doubles the ratio for the same reason.
      </p>
      <Ask q={questions51['ch05/5.1/q-decode']} />
    </>
  );
}
