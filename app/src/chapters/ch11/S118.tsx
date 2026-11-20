import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { fmt, realBackends, useReports, type BackendRun, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BOOK_DEVICE,
  ELEM_LABEL,
  WELL_OPTIMISED,
  WRITTEN_LABEL,
  bestRegisterTiled,
  blockSizes,
  ceiling,
  copyGBs,
  intensityLimit,
  kernelRates,
  labRows,
  measuredBandwidth,
  scanBytes,
  threshold,
  toppBytesPerElement,
  type Elem,
  type KernelRate,
  type Row,
  type RunLike,
  type Written,
} from '../../sim/scantraffic';
import { NoteGroup as Notes } from './NoteGroup';

// Section 11.8: a scan's operations per byte, the bandwidth ceiling in elements per second,
// and the 80% yardstick. Every number comes from src/sim/scantraffic.ts; the reader's own
// figures come from the lab benchmarks through useReports().

const LAB2 = 'ch11/lab2_';
const fmtG = (elementsPerSec: number) => fmt(elementsPerSec / 1e9);
const pct = (x: number) => `${Math.round(x * 100)}%`;


// ---------------------------------------------------------------- devices
interface Device {
  key: string;
  label: string;
  /** GB/s */
  bw: number;
  /** FLOP/B, when known */
  threshold?: number;
  note: string;
  /** Lab 11.2's rows on this backend, if any. */
  rows: Row[];
}
const BOOK: Device = {
  key: 'book',
  label: "The book's example (H100 SXM, 3.35 TB/s)",
  bw: BOOK_DEVICE.bandwidthGBs,
  threshold: threshold(BOOK_DEVICE.tflopsNvidia, BOOK_DEVICE.bandwidthGBs),
  note: "The book's example device, p. 272: 3.35 TB/s and a threshold of 20 FLOP/B.",
  rows: [],
};

/** This machine's backends with a measured copy rate (lab 11.2's copy, else lab 11.1's, else lab 5.1's probe). */
function measuredDevices(reports: Reports): Device[] {
  return realBackends(reports).flatMap((b) => {
    const run = reports[b] as BackendRun & RunLike;
    const m = measuredBandwidth(run);
    if (!m) return [];
    return [
      {
        key: b,
        label: `Yours: ${String(run.device.name ?? b)} (${b})`,
        bw: m.gbs,
        threshold: m.threshold,
        note: `Measured on ${run.generated.slice(0, 10)} by ${m.source}: ${fmt(m.gbs)} GB/s. That is what a copy achieves here, not the vendor's peak.`,
        rows: labRows(run, LAB2),
      },
    ];
  });
}

// ---------------------------------------------------------------- the share-of-copy chart
function ShareChart({ rates }: { rates: KernelRate[] }) {
  const W = 760;
  const ROW = 22;
  const L = 250;
  const top = 24;
  const H = top + rates.length * ROW + 10;
  const max = Math.max(1.1, ...rates.map((r) => r.share)) * 1.02;
  const X = (s: number) => L + (s / max) * (W - L - 16);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Each kernel's rate as a share of the copy rate" style={{ width: '100%', maxWidth: W, minWidth: 640, display: 'block' }}>
      <line x1={X(WELL_OPTIMISED)} x2={X(WELL_OPTIMISED)} y1={top - 6} y2={H - 6} stroke="var(--ink)" strokeDasharray="4 3" />
      <text x={X(WELL_OPTIMISED)} y={top - 10} fontSize="11" textAnchor="middle">80%</text>
      <line x1={X(1)} x2={X(1)} y1={top - 6} y2={H - 6} stroke="var(--ink-3)" />
      <text x={X(1)} y={top - 10} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>copy</text>
      {rates.map((r, i) => {
        const y = top + i * ROW;
        const good = r.share >= WELL_OPTIMISED;
        return (
          <g key={i} data-share={r.share.toFixed(4)} data-label={r.label}>
            <text x={L - 8} y={y + 13} fontSize="11.5" textAnchor="end">{r.label.replace(/, \d+ threads$/, '')}</text>
            <rect x={L} y={y + 3} width={Math.max(1, X(r.share) - L)} height={ROW - 8} rx="2"
              fill={good ? 'var(--ok)' : 'var(--ink-3)'} opacity={good ? 0.85 : 0.6} />
            <text x={X(r.share) + 5} y={y + 13} fontSize="11">{pct(r.share)}</text>
          </g>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------- explorer: the scan ceiling
function CeilingExplorer() {
  const { reports } = useReports();
  const devices = useMemo(() => [BOOK, ...measuredDevices(reports)], [reports]);
  const [key, setKey] = useState('book');
  const [custom, setCustom] = useState(800);
  const [elem, setElem] = useState<Elem>('float');
  const [written, setWritten] = useState<Written>('scan');
  const dev: Device =
    key === 'custom'
      ? { key: 'custom', label: 'Custom', bw: custom, note: 'A bandwidth you typed.', rows: [] }
      : (devices.find((d) => d.key === key) ?? BOOK);

  const bytes = scanBytes(elem, written);
  const opb = intensityLimit(bytes.total);
  const cap = ceiling(dev.bw, bytes.total);
  const copy = copyGBs(dev.rows);
  const sizes = blockSizes(dev.rows);
  const [blockPick, setBlock] = useState<number | null>(null);
  const block = blockPick != null && sizes.includes(blockPick) ? blockPick : (sizes[sizes.length - 1] ?? null);
  const rows = block == null ? dev.rows : dev.rows.filter((r) => r.block === block);
  const rates = copy ? kernelRates(rows, copy) : [];
  const best = bestRegisterTiled(rates);

  return (
    <Bench
      title="The scan ceiling"
      task="Find the ceiling for floats on the book's example, then switch to half precision and to your own device."
      foot={
        <>
          Ideal traffic: every element loaded once and every result stored once, nothing else. Operations are the scan's
          additions, N − 1 for N elements; for large N that is 1 per element. {dev.note}
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Device
          <select value={key} onChange={(e) => setKey(e.target.value)} aria-label="device">
            {devices.map((d) => <option key={d.key} value={d.key}>{d.label}</option>)}
            <option value="custom">Custom bandwidth</option>
          </select>
        </label>
        {key === 'custom' ? (
          <label className="field">
            Bandwidth (GB/s)
            <input type="number" min="1" value={custom} aria-label="bandwidth" onChange={(e) => setCustom(Math.max(1, +e.target.value || 1))} />
          </label>
        ) : null}
        <label className="field">
          Element type
          <span className="seg" role="group" aria-label="Element type">
            {(['float', 'half'] as Elem[]).map((e) => (
              <button key={e} aria-pressed={elem === e} onClick={() => setElem(e)}>{ELEM_LABEL[e]}</button>
            ))}
          </span>
        </label>
        <label className="field">
          What the kernel writes
          <span className="seg" role="group" aria-label="What is written">
            {(['scan', 'scan+mask'] as Written[]).map((w) => (
              <button key={w} aria-pressed={written === w} onClick={() => setWritten(w)}>{WRITTEN_LABEL[w]}</button>
            ))}
          </span>
        </label>
      </div>
      <div className="cols">
        <div className="panel" data-testid="ceiling">
          <p style={{ margin: 0 }}>
            Per element: <b data-testid="load">{bytes.load}</b> bytes in, <b data-testid="store">{bytes.store}</b> bytes out,{' '}
            <b data-testid="bytes">{bytes.total}</b> in all.
          </p>
          <p style={{ margin: '0.4rem 0 0' }}>
            Intensity: 1 addition per {bytes.total} bytes = <b data-testid="opb">{fmt(opb)}</b> OP/B.
          </p>
          <p style={{ fontSize: '1.5rem', margin: '0.6rem 0 0.2rem', fontWeight: 600 }}>
            <span data-testid="cap">{fmtG(cap)}</span> × 10⁹ elements/s
          </p>
          <p className="muted" style={{ margin: 0 }}>
            {fmt(dev.bw)} GB/s ÷ {bytes.total} bytes per element: the most this device can scan per second, however the kernel
            is written.
          </p>
          {dev.threshold ? (
            <p style={{ margin: '0.6rem 0 0' }} data-testid="regime">
              <span className="chip global">memory-bound</span> The device's threshold is {fmt(dev.threshold)} FLOP/B{key === 'book' ? '' : ' (lab 5.1)'}; the scan
              supplies {fmt(opb)}, about <b>{fmt(dev.threshold / opb)}</b> times less.
            </p>
          ) : null}
        </div>
        <div className="panel" data-testid="measured">
          {key === 'book' || key === 'custom' ? (
            <p style={{ margin: 0 }}>
              Pick your own device to see lab 11.2's kernels against your copy kernel. A published peak tells you what the
              memory could do; your copy kernel tells you what a kernel with nothing else to do actually gets.
            </p>
          ) : rates.length ? (
            <p style={{ margin: 0 }}>
              <b>Lab 11.2 on this device</b>: below, each kernel's rate as a share of the copy kernel's at the same block size
              (8 bytes per element), against the book's 80% line. Best at {block} threads:{' '}
              <b>{best!.rate.label.replace(/, \d+ threads$/, '')}</b>, {pct(best!.rate.share)}.
            </p>
          ) : (
            <p style={{ margin: 0 }}>
              No lab 11.2 benchmark on this device yet. Run <code>python3 labs/pmpp.py run 11.2 --solution --bench</code>, then
              reload: each of its kernels appears here as a share of the copy rate, against the 80% line.
            </p>
          )}
        </div>
      </div>
      {key !== 'book' && key !== 'custom' && rates.length ? (
        <div style={{ marginTop: '0.9rem' }} data-testid="chart">
          {sizes.length > 1 ? (
            <div className="row" style={{ marginBottom: '0.5rem' }}>
              <span className="seg" role="group" aria-label="Block size">
                {sizes.map((b) => (
                  <button key={b} aria-pressed={b === block} onClick={() => setBlock(b)}>{b} threads</button>
                ))}
              </span>
            </div>
          ) : null}
          <ShareChart rates={rates} />
        </div>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- measured reveal
function CopyReveal({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = realBackends(reports).flatMap((b) => {
    const rows = labRows(reports[b] as BackendRun & RunLike, LAB2);
    const copy = copyGBs(rows);
    const best = copy ? bestRegisterTiled(kernelRates(rows, copy)) : undefined;
    return best ? [{ b, device: String(reports[b].device.name ?? b), ...best }] : [];
  });
  if (!found.length)
    return (
      <p>
        No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 11.2 --solution --bench</code> on a machine with
        a GPU, then reload: this box will show the best register-tiled rate as a share of your copy kernel's.
      </p>
    );
  return (
    <>
      {found.map((f) => (
        <p key={f.b} data-backend={f.b} data-share={f.rate.share.toFixed(4)}>
          On <b>{f.device}</b> ({f.b}) the best {f.marked ? 'register-tiled kernel' : 'kernel'}, <b>{f.rate.label}</b>, reached{' '}
          <b>{pct(f.rate.share)}</b> of the copy rate, {Math.abs(f.rate.share * 100 - predicted) <= 5 ? 'within 5 points of your prediction' : f.rate.share * 100 > predicted ? `${Math.round(f.rate.share * 100 - predicted)} points above your prediction` : `${Math.round(predicted - f.rate.share * 100)} points below your prediction`}
          {f.rate.share >= WELL_OPTIMISED ? ', past the book\'s 80% line.' : ', short of the book\'s 80% line.'}
        </p>
      ))}
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS118: Record<string, Question> = {
  'ch11/11.8/q-half': {
    kind: 'numeric',
    id: 'ch11/11.8/q-half',
    prompt: (
      <>
        A scan reads half-precision values, adds them in a float accumulator, and writes half-precision results. What is its
        intensity in OP/B, for large N?
      </>
    ),
    answer: 0.25,
    tol: 0.02,
    unit: 'OP/B',
    explain: (
      <p>
        N − 1 additions over 2N bytes read and 2N written: about 1/4. The accumulator's width moves no bytes; only what is
        loaded from and stored to global memory counts. Halving the element size doubles the intensity and the ceiling in
        elements per second, and leaves the scan just as memory-bound: 0.25 is still 80 times below the book's threshold.
      </p>
    ),
    source: 'the §11.8 count (p. 272) applied to 2-byte elements; ours',
  },
  'ch11/11.8/q-ceiling': {
    kind: 'numeric',
    id: 'ch11/11.8/q-ceiling',
    prompt: (
      <>
        On some device a plain copy kernel measures 800 GB/s. A single-pass kernel scans floats there. What is its ceiling, in
        billions of elements per second?
      </>
    ),
    answer: 100,
    tol: 0.01,
    unit: '× 10⁹ elements/s',
    explain: (
      <p>
        Each element costs at least 8 bytes: 4 read, 4 written. 800 × 10⁹ bytes per second ÷ 8 = 100 × 10⁹ elements per
        second. Using the copy kernel's measured rate instead of a published peak gives a ceiling this device actually
        reaches with a kernel that does nothing else.
      </p>
    ),
    source: 'ch. 11 §11.8, p. 272 (the method); the 800 GB/s device is made up',
  },
  'ch11/11.8/q-topp-bytes': {
    kind: 'numeric',
    id: 'ch11/11.8/q-topp-bytes',
    prompt: (
      <>
        The fused top-p kernel of lab 11.6 reads one float probability per element and writes a float exclusive mass and a
        one-byte keep flag per element. How many bytes per element does it move?
      </>
    ),
    answer: toppBytesPerElement(),
    unit: 'bytes',
    explain: (
      <p>
        4 in, 4 + 1 out: 9 bytes. Its ceiling is the bandwidth over 9, a little below a plain scan's bandwidth over 8. The
        alternative, separate kernels for the scan and the threshold, would read the probabilities twice and write the mass
        and read it back: every pass over the row adds bytes the fused kernel never moves.
      </p>
    ),
    source: 'lab 11.6 (ours); the counting is §11.8, p. 272',
  },
  'ch11/11.8/q-80': {
    kind: 'choice',
    id: 'ch11/11.8/q-80',
    prompt: (
      <>
        Your scan kernel's profile says 85% of the published bandwidth. Looking closer, it reads its whole input twice. Is it
        well optimised by the book's rule?
      </>
    ),
    options: [
      { label: 'Yes: anything above 80% counts.', why: 'The rule has two parts. Read p. 273 again: above 80% and no redundant global accesses.' },
      {
        label: 'No: the rule also requires no redundant global accesses. It moves 12 bytes per element where 8 suffice, so its useful rate is about two thirds of what the measurement suggests.',
        correct: true,
      },
      { label: 'No: a scan can never pass 80%, so the measurement must be wrong.', why: 'The book says Fig. 11.13 can plausibly exceed 80% (p. 273).' },
      { label: 'It cannot be judged without knowing the device\'s FLOP/s.', why: 'At 1/8 OP/B the scan is memory-bound on any device the book considers; only bytes matter.' },
    ],
    explain: (
      <p>
        Bandwidth measures bytes moved, not work done. Reading the input twice turns 8 bytes per element into 12, so 85% of
        the bandwidth delivers 85% × 8/12 ≈ 57% of the elements per second a kernel touching each location once could reach.
        The second half of the rule, no redundant accesses, is what stops a wasteful kernel from looking good.
      </p>
    ),
    source: 'ch. 11 §11.8, pp. 272–273',
  },
};

export function S118() {
  return (
    <>
      <p>
        Section 11.7 left you with a block scan that touches shared memory once per element on the way in and once on the way
        out. Before going further, ask what the fastest possible scan would be. For a scan the answer takes three numbers and
        one division.
      </p>
      <Source section="11.8" pages="272" />

      <h2>Count the bytes</h2>
      <p>
        Scan N floats. The kernel has to read every input, 4N bytes, and write every result, another 4N bytes. In between it
        does N − 1 additions. Operations per byte of global traffic: (N − 1) / 8N, which for any real N is 1/8 OP/B. That is
        a property of the computation, not of how cleverly it is written: no kernel can do fewer additions or move fewer
        bytes.
      </p>
      <p>
        Compare it with a device. The book's example GPU does 66.9 TFLOPS against 3.35 TB/s, a threshold of 20 FLOP/B (the
        same calculation as section 5.1). A scan at 0.125 is 160 times short of it. The arithmetic units will spend almost
        all their time waiting, so for a scan the bandwidth is the whole story.
      </p>
      <Today checked="2026-11-20 against nvidia.com (notes/today/ch11-facts.md A1; ch05 A8)">
        <p>
          NVIDIA prints FP32 for the H100 SXM as 67 TFLOPS, not 66.9, and confirms 3.35 TB/s. The threshold is 20 FLOP/B either
          way, and every ceiling below is unchanged.
        </p>
      </Today>

      <h2>The ceiling</h2>
      <p>
        For a memory-bound kernel, peak bandwidth sets peak performance. A scan must move at least 8 bytes per float, so the
        most elements it can process per second is the bandwidth divided by 8. On the book's example, 3.35 × 10¹² / 8 = 419 ×
        10⁹ elements per second. Your device has its own figure, and the honest way to get it is to divide what a plain copy
        kernel achieves there, not the number on the vendor's page.
      </p>
      <Source section="11.8" pages="272" />
      <CeilingExplorer />
      <p>
        Two things to take from the explorer. The intensity never moves far: halving the element size doubles it to 1/4, and a
        one-byte flag lowers it to 1/9, but every option stays far below the book's threshold and your own device's, so the
        regime never changes. What does move is the ceiling in elements per second, in exact proportion to the bytes per element. For a
        memory-bound kernel, a byte saved is time saved.
      </p>
      <Simplification>
        <p>
          The explorer charges every byte to global memory and nothing else: no caches, no launch overhead, no partial last
          block. It is a ceiling, not a prediction.
        </p>
      </Simplification>
      <Ask q={questionsS118['ch11/11.8/q-half']} />
      <Ask q={questionsS118['ch11/11.8/q-ceiling']} />

      <h2>How close is close enough</h2>
      <p>
        Reaching the ceiling would mean touching every global location exactly once and hiding every addition, every barrier
        and every shared-memory access behind the global traffic. The book says that does not happen: even a kernel that only
        copies does not reach 100% of peak bandwidth. Its yardstick instead: a memory-bound kernel that reaches more than 80%
        of the bandwidth, <em>with no redundant global accesses</em>, counts as well optimised.
      </p>
      <Notes>
      <Source section="11.8" pages="272–273" />
      <Today checked="2026-11-20 against the Nsight Compute documentation v2026.4.0 (notes/today/ch11-facts.md A2)">
        <p>
          NVIDIA's profiler draws the same line: its triage guide calls throughput "Above ~80%: high (near-limit regime)". No
          vendor source was found for the book's sentence that a copy cannot reach 100%; treat that as the book's.
        </p>
      </Today>
      </Notes>
      <p>
        That suggests measuring twice: against the published peak, and against your own copy kernel on the same device and
        the same size. The second tells you how much of the gap is the scan's fault and how much is simply what the memory
        system delivers to any kernel. Lab 11.2 times a copy beside every scan it runs, for this reason.
      </p>
      <Beyond>
        <p>Comparing with your own copy kernel as well as the peak is our suggestion, not the book's.</p>
      </Beyond>
      <Predict
        id="ch11/11.8/p-copy"
        prompt="What share of your own copy kernel's bandwidth will lab 11.2's register-tiled kernel reach at its best coarsening factor?"
        unit="%"
      >
        {(v) => <CopyReveal predicted={v} />}
      </Predict>
      <Ask q={questionsS118['ch11/11.8/q-80']} />

      <h2>What the block scan leaves undone</h2>
      <p>
        The register-tiled kernel of Fig. 11.13 reads each element once and writes it once, so by the book's judgement it can
        plausibly pass 80%. But it only scans each block's segment on its own. Turning those segments into one scan of the
        whole array needs every block to learn the total of all the blocks before it, and that costs more global accesses and
        more synchronisation. How many more is the subject of <a href={href('ch11', '11.9')}>section 11.9</a>, and it is
        counted in exactly these units: bytes per element.
      </p>
      <Source section="11.8" pages="273" />

      <h2>Underneath a sampler</h2>
      <p>
        A cumulative sum over one row of probabilities, the step at the heart of top-p sampling, is this same computation: one
        addition per element, 1/8 OP/B in float. Run it as separate kernels for softmax normalisation, cumulative sum and
        threshold, and each kernel reads the row and writes a result the next one reads back. Fused into one pass, the row is
        read once and the mass and the keep flags are written once: 9 bytes per element in lab 11.6. That is section 11.9's
        argument, load once and store once, applied to milestone M5.
      </p>
      <Beyond>
        <p>
          The book does not connect scan to sampling. The mapping and lab 11.6 are ours; the counting is the book's (p. 272).
        </p>
      </Beyond>
      <Ask q={questionsS118['ch11/11.8/q-topp-bytes']} />
    </>
  );
}
