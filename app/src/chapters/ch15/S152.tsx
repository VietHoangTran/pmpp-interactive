import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { BOOK_THRESHOLD, ELEMENT_BYTES, gridFetches, regime, tilePair } from '../../sim/gemmreuse';
import { ReuseScene } from './ReuseScene';

// ---------------------------------------------------------------- questions
export const questionsS152: Record<string, Question> = {
  'ch15/15.2/q-bk': {
    kind: 'choice',
    id: 'ch15/15.2/q-bk',
    prompt: <>In the 128 × 128 kernel the tile depth bK doubles from 8 to 16. What changes?</>,
    options: [
      {
        label: 'Intensity is unchanged; one tile pair\'s shared memory doubles to 16 KB; half as many tile pairs, and barriers, cover K.',
        correct: true,
      },
      { label: 'Intensity doubles: each loaded element now feeds twice as many multiply-adds, while shared memory per tile pair stays at 8 KB.', why: 'Operations and bytes per tile pair both double, so their ratio does not move, and the pair now takes 16 KB.' },
      { label: 'Intensity halves, because each tile pair now loads twice as many bytes into shared memory.', why: 'Operations and bytes per tile pair both double, so their ratio does not move.' },
      { label: 'Nothing that matters: intensity, shared memory per pair and the number of barriers all stay the same.', why: 'The ratio does not, but each pair now takes twice the shared memory, and half as many pairs cover K.' },
    ],
    explain: (
      <p>
        One tile pair is 4 × 8 × (128 + 128) = 8,192 bytes at bK = 8 and 16,384 at bK = 16, and the shared memory has to
        hold it. K is covered in K / bK pairs, each with its two barriers. Section 15.7 uses exactly this: since bK buys no
        intensity, the book keeps it small to save shared memory.
      </p>
    ),
    source: 'ch. 15 §15.2, p. 351; §15.7, p. 363',
  },
  'ch15/15.2/q-rect': {
    kind: 'numeric',
    id: 'ch15/15.2/q-rect',
    prompt: <>What is the intensity of a 128 × 64 output tile with 4-byte elements?</>,
    answer: 21.33,
    tol: 0.02,
    unit: 'FLOP/B',
    explain: (
      <p>
        0.5 × 128 × 64 / (128 + 64) = 4096 / 192 ≈ 21.3 FLOP/B: just above the book's example threshold of 20. A
        rectangle sits between its square neighbours, 64 × 64 (16) and 128 × 128 (32), closer to the smaller side.
      </p>
    ),
    source: 'ch. 15 §15.2, p. 351',
  },
  'ch15/15.2/q-decode': {
    kind: 'choice',
    id: 'ch15/15.2/q-decode',
    prompt: (
      <>
        At batch size 1, a linear layer's product has m = 1: one row of output per tile. Which change raises its
        intensity?
      </>
    ),
    options: [
      { label: 'Batching more rows, or fewer bytes per weight.', correct: true },
      { label: 'A wider output tile, so that one block covers more of the layer\'s outputs.', why: 'With m = 1 the intensity is 0.5 n / (n + 1), below 0.5 for every n. Widening approaches 0.5 and never passes it.' },
      { label: 'A deeper tile, larger bK, so that each load brings in more of the inner dimension.', why: 'k cancels: it changes operations and bytes by the same factor.' },
      { label: 'More threads per block, so that more multiply-adds are in flight at once.', why: 'Threads change who does the work, not how many bytes each operation needs.' },
    ],
    explain: (
      <p>
        One row of output uses each weight once: two operations per weight fetched, 0.5 FLOP/B in float32, whatever the
        kernel. Batching makes a fetched weight serve more rows (m grows); quantisation makes each weight fewer bytes.
        llama.cpp switches to matrix-vector kernels for batches up to 8.
      </p>
    ),
    source: 'our derivation from the formula on p. 351; llama.cpp from ch15-facts D1',
  },
  'ch15/15.2/q-batch59': {
    kind: 'numeric',
    id: 'ch15/15.2/q-batch59',
    prompt: (
      <>
        Output tiles are 128 columns wide, and their height is the batch size b (b ≤ 128), in float32. What is the
        smallest b that reaches the book's example threshold of 20 FLOP/B?
      </>
    ),
    answer: 59,
    unit: 'rows',
    explain: (
      <p>
        0.5 × 128 b / (b + 128) = 64 b / (b + 128) ≥ 20 gives 44 b ≥ 2560, so b ≥ 58.2: 59. Check it in the explorer:
        58 × 128 gives 19.96 and 59 × 128 gives 20.19.
      </p>
    ),
    source: 'our arithmetic with the formula on p. 351',
  },
  'ch15/15.2/q-fetches': {
    kind: 'numeric',
    id: 'ch15/15.2/q-fetches',
    prompt: (
      <>
        M = N = K = 4096, computed with 128 × 128 output tiles. In the model (no caches), how many times is each element
        of A fetched from global memory?
      </>
    ),
    answer: 32,
    unit: 'times',
    explain: (
      <p>
        Every block in a row of C reads that row band of A, and a row of C holds ⌈N / bN⌉ = 4096 / 128 = 32 blocks.
        With 32-wide tiles it would be 128 times: four times the traffic for the same operations.
      </p>
    ),
    source: 'ch. 15 §15.2, pp. 351–352; the count is ours',
  },
};

// ---------------------------------------------------------------- thresholds
interface Threshold {
  key: string;
  label: string;
  value: number;
  note: string;
}
const BOOK: Threshold = {
  key: 'book',
  label: "The book's example (H100, from chapter 5)",
  value: BOOK_THRESHOLD,
  note: "The book's example device: about 20 FLOP/B (chapter 5).",
};

function measuredThresholds(reports: Reports): Threshold[] {
  return realBackends(reports).flatMap((b) => {
    const r = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
    const peak = num(r?.compute_gflops);
    const bw = num(r?.bandwidth_gb_per_s);
    if (!peak || !bw) return [];
    return [{
      key: b,
      label: `Yours: ${String(reports[b].device.name ?? b)} (${b})`,
      value: peak / bw,
      note: `Measured by lab 5.1 on ${reports[b].generated.slice(0, 10)}: ${fmt(peak)} GFLOP/s ÷ ${fmt(bw)} GB/s, two probe kernels, not vendor peaks.`,
    }];
  });
}

function useThreshold() {
  const { reports } = useReports();
  const options = useMemo(() => [BOOK, ...measuredThresholds(reports)], [reports]);
  const [key, setKey] = useState('book');
  const [custom, setCustom] = useState(10);
  const th: Threshold = key === 'custom'
    ? { key, label: 'Custom', value: custom, note: 'A number you typed.' }
    : (options.find((o) => o.key === key) ?? BOOK);
  const picker = (
    <label className="field">
      Threshold
      <select value={key} aria-label="threshold" onChange={(e) => setKey(e.target.value)}>
        {options.map((o) => (
          <option key={o.key} value={o.key}>{o.label}</option>
        ))}
        <option value="custom">Custom number</option>
      </select>
      {key === 'custom' ? (
        <input type="number" min="0.1" step="any" value={custom} aria-label="custom threshold"
          onChange={(e) => setCustom(Math.max(0.1, +e.target.value || 0.1))} />
      ) : null}
    </label>
  );
  return { th, picker, measuredCount: options.length - 1 };
}

const RegimeChip = ({ fpb, threshold }: { fpb: number; threshold: number }) =>
  regime(fpb, threshold) === 'memory-bound' ? (
    <span className="chip global" data-regime="memory-bound">memory-bound</span>
  ) : (
    <span className="chip reg" data-regime="compute-bound">compute-bound</span>
  );

const bytesLabel = (b: number) => (b === 4 ? '4 (float32)' : b === 2 ? '2 (16-bit)' : b === 1 ? '1 (8-bit)' : '0.5 (4-bit)');

// ---------------------------------------------------------------- explorer 1: one output tile
function TileDrawing({ m, n, k }: { m: number; n: number; k: number }) {
  // Fig. 15.1's L layout, to scale: A's tile left of C, B's tile above it.
  const s = 0.62; // px per element
  const pad = 8;
  const kw = Math.max(2, k * s);
  const W = pad + 64 * s + 6 + 256 * s + pad;
  const H = pad + 64 * s + 6 + 256 * s + pad;
  const cx = pad + 64 * s + 6;
  const cy = pad + 64 * s + 6;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="The tile pair and the output tile, to scale" data-layout="narrow"
      style={{ width: '100%', maxWidth: 260, display: 'block' }}>
      <rect x={cx - 6 - kw} y={cy} width={kw} height={Math.max(2, m * s)} fill="var(--shared-wash)" stroke="var(--shared)" data-tile="A" />
      <rect x={cx} y={cy - 6 - kw} width={Math.max(2, n * s)} height={kw} fill="var(--shared-wash)" stroke="var(--shared)" data-tile="B" />
      <rect x={cx} y={cy} width={Math.max(2, n * s)} height={Math.max(2, m * s)} fill="var(--reg-wash)" stroke="var(--reg)" data-tile="C" />
      <text x={cx + 4} y={cy + Math.max(2, m * s) + 14} fontSize="11" style={{ fill: 'var(--reg)' }}>C tile {m} × {n}</text>
    </svg>
  );
}

const KCOVER = 4096;

function IntensityBench() {
  const { th, picker } = useThreshold();
  // Starts between the book's two examples, so reaching each of them is the reader's move.
  const [m, setM] = useState(64);
  const [n, setN] = useState(64);
  const [k, setK] = useState(8);
  const [bytes, setBytes] = useState<number>(4);
  // Checkpoints: which of the section's discoveries the reader has reached.
  const [kSeen, setKSeen] = useState<{ key: string; ks: number[] }>({ key: '64,64,4', ks: [8] });
  const [hit, setHit] = useState<Record<string, boolean>>({});
  const t = tilePair(m, n, k, bytes);
  const shapeKey = `${m},${n},${bytes}`;

  const note = (nm: number, nn: number, nk: number, nb: number) => {
    const key = `${nm},${nn},${nb}`;
    const ks = key === kSeen.key ? (kSeen.ks.includes(nk) ? kSeen.ks : [...kSeen.ks, nk]) : [nk];
    setKSeen({ key, ks });
    setHit((h) => ({
      ...h,
      eight: h.eight || (nm === 32 && nn === 32 && nb === 4),
      thirtytwo: h.thirtytwo || (nm === 128 && nn === 128 && nb === 4),
      kfree: h.kfree || ks.length >= 3,
      one: h.one || nm === 1,
    }));
  };
  const set = (what: 'm' | 'n' | 'k', v: number) => {
    const nm = what === 'm' ? v : m;
    const nn = what === 'n' ? v : n;
    const nk = what === 'k' ? v : k;
    if (what === 'm') setM(v);
    if (what === 'n') setN(v);
    if (what === 'k') setK(v);
    note(nm, nn, nk, bytes);
  };
  const setB = (b: number) => {
    setBytes(b);
    note(m, n, k, b);
  };
  const checks: [string, string][] = [
    ['eight', 'Reproduce the book\'s 8 FLOP/B (32 × 32, float32).'],
    ['thirtytwo', 'Reproduce the book\'s 32 FLOP/B (128 × 128, float32).'],
    ['kfree', 'Try three depths k for the same tile, and watch the intensity.'],
    ['one', 'Set m = 1: one row of output, as in decoding one token.'],
  ];
  const range = (label: string, what: 'm' | 'n' | 'k', v: number, max: number) => (
    <label className="field">
      {label} = {v}
      <input type="range" min="1" max={max} value={v} aria-label={what} onChange={(e) => set(what, +e.target.value)} />
    </label>
  );
  return (
    <Bench
      title="Intensity of an output tile"
      task="Reproduce the book's two numbers, then find what k does."
      foot={
        <>
          The model of §15.2: only the tile loads from global memory are counted, the output tile's single write is not,
          and there are no caches. The bytes setting shrinks both input tiles; weight-only quantisation shrinks only the
          weights' tile, which in a linear layer is the n side. Drawing to scale except that tiles thinner than 2 pixels
          are drawn at 2. {th.note}
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        {range('m, output tile height', 'm', m, 256)}
        {range('n, output tile width', 'n', n, 256)}
        {range('k, tile depth', 'k', k, 64)}
        <label className="field">
          Bytes per element
          <span className="seg">
            {ELEMENT_BYTES.map((b) => (
              <button key={b} aria-pressed={bytes === b} onClick={() => setB(b)}>{bytesLabel(b)}</button>
            ))}
          </span>
        </label>
        {picker}
      </div>
      <div className="cols">
        <div className="panel">
          <TileDrawing m={m} n={n} k={k} />
          <ul style={{ margin: '0.6rem 0 0', paddingLeft: '1.1rem' }} data-checkpoints>
            {checks.map(([id, text]) => (
              <li key={id} data-check={id} data-done={hit[id] ? 1 : 0} style={{ color: hit[id] ? 'var(--ok)' : undefined }}>
                {hit[id] ? '✓ ' : ''}{text}
              </li>
            ))}
          </ul>
        </div>
        <div className="panel">
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <tr><td>Operations per tile pair, 2·m·n·k</td><td className="num" data-out="ops">{fmt(t.ops, 6)}</td></tr>
              <tr><td>Bytes per tile pair, {bytes}·k·(m + n)</td><td className="num" data-out="bytes">{fmt(t.traffic, 6)}</td></tr>
              <tr><td>Shared memory to hold the pair</td><td className="num" data-out="smem">{fmt(t.traffic / 1024, 4)} KB</td></tr>
              <tr><td>Tile pairs to cover K = {KCOVER}</td><td className="num" data-out="pairs">{Math.ceil(KCOVER / k)}</td></tr>
            </tbody>
          </table>
          <p style={{ fontSize: '1.6rem', margin: '0.7rem 0 0.2rem', fontWeight: 600 }} data-out="intensity" data-value={t.intensity}>
            {fmt(t.intensity, 4)} FLOP/B
          </p>
          <p className="muted" style={{ margin: 0 }} data-shape={shapeKey}>
            <RegimeChip fpb={t.intensity} threshold={th.value} /> against a threshold of {fmt(th.value)} FLOP/B.
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer 2: the whole grid
const GRID_PRESETS: { label: string; M: number; N: number; K: number }[] = [
  { label: 'square, 4096', M: 4096, N: 4096, K: 4096 },
  { label: '1024 → 4096 layer, batch 1', M: 1, N: 4096, K: 1024 },
  { label: '4096 → 4096 layer, batch 8', M: 8, N: 4096, K: 4096 },
  { label: '1024 → 4096 layer, batch 256', M: 256, N: 4096, K: 1024 },
];
const TILES = [32, 64, 128, 256];

function GridBench() {
  const [M, setM] = useState(4096);
  const [N, setN] = useState(4096);
  const [K, setK] = useState(4096);
  const [bM, setBM] = useState(128);
  const [bN, setBN] = useState(128);
  const g = gridFetches(M, N, K, bM, bN);
  const numIn = (label: string, v: number, set: (x: number) => void, key: string) => (
    <label className="field">
      {label}
      <input type="number" min="1" max="65536" value={v} aria-label={key}
        onChange={(e) => set(Math.max(1, Math.min(65536, Math.round(+e.target.value) || 1)))} />
    </label>
  );
  const tileSel = (label: string, v: number, set: (x: number) => void, key: string) => (
    <label className="field">
      {label}
      <select value={v} aria-label={key} onChange={(e) => set(+e.target.value)}>
        {TILES.map((t) => <option key={t} value={t}>{t}</option>)}
      </select>
    </label>
  );
  const W = 260;
  return (
    <Bench
      title="Fetches across the grid"
      task="Find how many times each input element is fetched, then shrink the batch."
      foot="Counts follow from the tile layout alone: every block reads its whole row band of A and column band of B. No caches, output writes not counted. 1 MB = 10⁶ bytes."
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        {numIn('M (rows of A and C)', M, setM, 'grid M')}
        {numIn('N (columns of B and C)', N, setN, 'grid N')}
        {numIn('K (inner)', K, setK, 'grid K')}
        {tileSel('bM', bM, setBM, 'bM')}
        {tileSel('bN', bN, setBN, 'bN')}
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', gap: '0.4rem' }}>
        <span className="ui muted">Presets:</span>
        {GRID_PRESETS.map((p) => (
          <button key={p.label} className="quiet" onClick={() => { setM(p.M); setN(p.N); setK(p.K); }}>{p.label}</button>
        ))}
      </div>
      <div className="cols">
        <div className="panel">
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <tr><td>Blocks</td><td className="num" data-out="blocks">{g.blocksM} × {g.blocksN}</td></tr>
              <tr><td>Fetches of each element of A, ⌈N / bN⌉</td><td className="num" data-out="afetch">{g.aFetches}</td></tr>
              <tr><td>Fetches of each element of B, ⌈M / bM⌉</td><td className="num" data-out="bfetch">{g.bFetches}</td></tr>
              <tr><td>Bytes requested (float32)</td><td className="num" data-out="requested">{fmt(g.requested / 1e6, 4)} MB</td></tr>
              <tr><td>Bytes if each element came once</td><td className="num" data-out="ideal">{fmt(g.ideal / 1e6, 4)} MB</td></tr>
            </tbody>
          </table>
        </div>
        <div className="panel">
          <svg viewBox={`0 0 ${W + 70} 70`} role="img" aria-label="Requested against needed bytes" style={{ width: '100%', maxWidth: 360, display: 'block' }}>
            <text x="0" y="12" fontSize="11">requested</text>
            <rect x="0" y="16" width={W} height="14" fill="var(--global)" />
            <text x="0" y="48" fontSize="11">needed</text>
            <rect x="0" y="52" width={Math.max(1, (W * g.ideal) / g.requested)} height="14" fill="var(--shared)" />
            <text x={W + 6} y="28" fontSize="11">× {fmt(g.requested / g.ideal)}</text>
          </svg>
          <p style={{ fontSize: '1.4rem', margin: '0.6rem 0 0.2rem', fontWeight: 600 }} data-out="grid-intensity" data-value={g.intensity}>
            {fmt(g.intensity, 4)} FLOP/B
          </p>
          <p className="muted" style={{ margin: 0 }}>
            2·M·N·K operations over the bytes requested: the whole kernel's intensity in this model.
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the page
function GatedExplorer() {
  const { progress } = useProgress();
  const committed = typeof progress.answers['ch15/15.2/p-intensity128']?.value === 'number';
  return committed ? (
    <IntensityBench />
  ) : (
    <Bench title="Intensity of an output tile" task="Opens once you have committed a prediction above.">
      <p style={{ margin: 0 }} data-gated>Commit your prediction for the 128 × 128 tile first; the explorer would give it away.</p>
    </Bench>
  );
}

export function S152() {
  return (
    <>
      <p>
        Chapter 5's tiles were square, with one output per thread, and its count of reuse assumed square tiles. To go
        further you need the count for any tile shape. It turns out that only one thing about the tile matters.
      </p>

      <h2>The setting</h2>
      <p>
        One thread block owns an m × n tile of C. That tile lives in the block's registers, spread over its threads, from
        the first step to the last. To fill it, the block walks along the inner dimension K in steps of k. At each step it
        loads an m × k tile of A and the matching k × n tile of B from global memory into shared memory, and every thread
        adds that pair's contribution to its outputs. The output tile goes to global memory once, at the end.
      </p>
      <Source section="15.2" pages="350–351" figs="Fig. 15.1" />

      <h2>Count it</h2>
      <p>
        For one tile pair: each of the m·n outputs gets k multiplications and k additions, so 2mnk operations. The block
        loads m·k + k·n floats, which is 4k(m + n) bytes. Divide one by the other and k appears on both sides:
      </p>
      <p style={{ textAlign: 'center', fontSize: '1.1rem' }}>
        2mnk / 4k(m + n) = <b>0.5 · m · n / (m + n)</b> FLOP/B
      </p>
      <p>
        The depth of a step cancels. Only the output tile's height and width are left, and the ratio grows with both.
      </p>
      <Source section="15.2" pages="351" />
      <Predict
        id="ch15/15.2/p-intensity128"
        prompt="What is the intensity of a 128 × 128 output tile with 4-byte elements?"
        unit="FLOP/B"
      >
        {(v) => (
          <p>
            0.5 × 16,384 / 256 = <b>32 FLOP/B</b>.{' '}
            {Math.abs(v - 32) < 0.5
              ? 'That is what you had.'
              : v > 32
                ? 'If you were higher, check the denominator: it is m + n, not m or n alone.'
                : 'If you were lower, check that the 4 bytes appear once: 2mnk over 4k(m + n).'}{' '}
            Four times the 8 FLOP/B of chapter 5's 32 × 32 tiles: the tile has sixteen times the outputs, while each step
            loads only four times the bytes.
          </p>
        )}
      </Predict>
      <GatedExplorer />

      <h2>Against a threshold</h2>
      <p>
        Chapter 5 drew the line between a kernel that starves for data and one that keeps its arithmetic busy: the
        device's peak operations per second divided by its peak bytes per second. On the book's example device that line
        is at 20 FLOP/B. Chapter 5's 32 × 32 tiles give 8 and stay memory-bound there; 128 × 128 tiles give 32 and are
        compute-bound.
      </p>
      <Source section="15.2" pages="351" />
      <p>
        On your own device the line is somewhere else. If you have run lab 5.1, pick your device in the explorer's
        threshold menu (<a href={href('ch05', 'labs')}>chapter 5 labs</a>).
      </p>
      <Today checked="2026-11-20 (ch15-facts A1; ch05-facts A8)">
        <p>
          The book's device is an H100: NVIDIA prints 67 TFLOPS FP32 and 3.35 TB/s (SXM), 20.0 FLOP/B. The book's 66.9
          is in no NVIDIA source.
        </p>
      </Today>
      <Ask q={questionsS152['ch15/15.2/q-rect']} />
      <Ask q={questionsS152['ch15/15.2/q-bk']} />

      <h2>Where the reuse comes from</h2>
      <p>
        A bigger output tile raises intensity because blocks stop fetching the same data. Cut C into 32-wide tiles and
        the four neighbouring blocks along a row of C all need the same band of A: each fetches it on its own. Make the
        tile 128 wide and one block does what those four did, and fetches the band once. Making the tile taller does the
        same for the bands of B.
      </p>
      <Source section="15.2" pages="351–352" />
      <ReuseScene />
      <p>
        Over a whole matrix the count is simple. Every element of A is fetched once by each block in its row of C,
        ⌈N / bN⌉ times; every element of B once by each block in its column, ⌈M / bM⌉ times.
      </p>
      <GridBench />
      <p>
        The preset layers show the same thing from the other side: when M is the batch and the batch is small, the
        blocks run out of rows, and every weight is fetched for very little work.
      </p>
      <Simplification>
        <p>
          Only input tiles loaded from global memory count; the output's single write is left out, as in the book's
          ratio. No caches: a real GPU serves some repeated fetches from cache, so bigger tiles can gain far less than the
          model's ratio, as chapter 5's lab found on the machine this course was built on.
        </p>
      </Simplification>
      <Ask q={questionsS152['ch15/15.2/q-fetches']} />

      <h2>Batch one</h2>
      <p>
        Now put a linear layer through the formula. In a tiled kernel for y = x Wᵀ, an output tile can be no taller than
        the batch. Generating one token at a time, m = 1, and the intensity is 0.5 n / (n + 1): below 0.5 FLOP/B however
        wide the tile. No tile shape rescues it. Two things do: more rows per fetched weight, which is batching, and
        fewer bytes per weight, which is quantisation. The explorer's bytes setting shrinks both inputs, so 2-byte
        elements double every intensity. Quantising only the weights shrinks only the k·n term of the bytes. At m = 1
        that term is nearly all of them, so the gain is nearly the same; as m grows, the activations' k·m term, still
        4 bytes an element, takes back a growing share.
      </p>
      <Beyond>
        <p>
          The m = 1 bound is our derivation from the book's formula. llama.cpp uses matrix-vector kernels up to batch 8
          (<code>MMVQ_MAX_BATCH_SIZE</code>, <code>MMVF_MAX_BATCH_SIZE</code>), and PyTorch's code says gemv "is bw bound"
          (ch15-facts D1, D4). Quantised kernels are the subject of <a href={href('x2')}>module X2</a>.
        </p>
      </Beyond>
      <Ask q={questionsS152['ch15/15.2/q-decode']} />
      <Ask q={questionsS152['ch15/15.2/q-batch59']} />
    </>
  );
}
