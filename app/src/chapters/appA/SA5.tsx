import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { E2M1, E2M3, E3M2, E4M3, E5M2, maxFinite, minSubnormal, showBits, type Format } from '../../sim/minifloat';
import {
  ELEMENTS,
  ELEMENT_IDS,
  SCALE_RULES,
  WEIGHT_FORMATS,
  WEIGHT_FORMAT_IDS,
  bitsPerWeight,
  convert,
  decodeCeiling,
  elemGrid,
  namedFormat,
  opsPerByte,
  quantizeBlock,
  sampleBlock,
  weightBytes,
  type ElemId,
  type ScaleRule,
  type WeightFormat,
} from '../../sim/mxblock';
import { MxBlockScene } from './MxBlockScene';

const minus = (s: string) => s.replace(/-/g, '−');
const show = (v: number, d = 4) =>
  Number.isNaN(v) ? 'NaN' : !Number.isFinite(v) ? (v > 0 ? 'inf' : '−inf') : v === 0 ? (Object.is(v, -0) ? '−0' : '0') : minus(String(Number(v.toPrecision(d))));

// ---------------------------------------------------------------- decode ceiling
interface Hw {
  key: string;
  label: string;
  gbs: number;
  note: string;
}
// The book's example device (ch. 5, p. 94): used only as the book's example.
const BOOK_HW: Hw = { key: 'book', label: "The book's example (H100, 3.35 TB/s)", gbs: 3350, note: 'The bandwidth the book quotes for its example GPU in chapter 5 (p. 94).' };

const P_MIN = 0.5e9;
const P_MAX = 70e9;
const paramsAt = (s: number) => Number((P_MIN * (P_MAX / P_MIN) ** (s / 1000)).toPrecision(2));
const sliderFor = (p: number) => Math.round((1000 * Math.log(p / P_MIN)) / Math.log(P_MAX / P_MIN));

function DecodeCeilingBench() {
  const { reports } = useReports();
  const measured: Hw[] = useMemo(
    () =>
      realBackends(reports).flatMap((b) => {
        const r = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
        const v = num(r?.bandwidth_gb_per_s);
        return v
          ? [{ key: b, label: `Yours: ${String(reports[b].device.name ?? b)} (${b}), ${fmt(v)} GB/s`, gbs: v, note: `Measured by lab 5.1 on ${reports[b].generated.slice(0, 10)}: a copy kernel, not the vendor's peak.` }]
          : [];
      }),
    [reports],
  );
  const options = [BOOK_HW, ...measured];
  const [hwKey, setHwKey] = useState('book');
  const [custom, setCustom] = useState(500);
  const [s, setS] = useState(sliderFor(7e9));
  const [sel, setSel] = useState<WeightFormat>('mxfp4');
  const params = paramsAt(s);
  const hw: Hw = hwKey === 'custom' ? { key: 'custom', label: 'Custom', gbs: custom, note: 'A bandwidth you typed.' } : (options.find((o) => o.key === hwKey) ?? BOOK_HW);
  const rows = WEIGHT_FORMAT_IDS.map((f) => {
    const bits = bitsPerWeight(f);
    return { f, bits, bytes: weightBytes(params, bits), ai: opsPerByte(bits), ceil: decodeCeiling(params, bits, hw.gbs) };
  });
  const top = Math.max(...rows.map((r) => r.ceil));
  const bf16 = rows.find((r) => r.f === 'bf16')!;
  const cur = rows.find((r) => r.f === sel)!;
  return (
    <Bench
      title="Decode ceiling by weight format"
      task="Pick a model size and a device, then compare the formats. What sets the ratio between two rows?"
      foot={<>Ceiling = bandwidth ÷ bytes of weights per token, at batch 1. {hw.note} No device figure here is an expected value for yours.</>}
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Parameters: {fmt(params / 1e9)} billion
          <input type="range" min="0" max="1000" value={s} aria-label="parameters" onChange={(e) => setS(+e.target.value)} />
        </label>
        <label className="field">
          Device bandwidth
          <select value={hwKey} onChange={(e) => setHwKey(e.target.value)} aria-label="device">
            {options.map((o) => (
              <option key={o.key} value={o.key}>{o.label}</option>
            ))}
            <option value="custom">Custom</option>
          </select>
        </label>
        {hwKey === 'custom' ? (
          <label className="field">
            GB/s
            <input type="number" min="1" value={custom} onChange={(e) => setCustom(Math.max(1, +e.target.value || 1))} />
          </label>
        ) : null}
        <label className="field">
          Weight format
          <select value={sel} onChange={(e) => setSel(e.target.value as WeightFormat)} aria-label="weight format">
            {WEIGHT_FORMAT_IDS.map((f) => (
              <option key={f} value={f}>{WEIGHT_FORMATS[f].label}</option>
            ))}
          </select>
        </label>
      </div>
      <p className="ceiling-head" style={{ margin: '0 0 0.2rem', fontSize: '1.45rem', fontWeight: 600 }} data-ceiling={cur.ceil}>
        {fmt(cur.ceil)} tokens per second, at most
      </p>
      <p className="muted" style={{ margin: '0 0 0.8rem' }}>
        {WEIGHT_FORMATS[sel].label}: {fmt(cur.bits)} bits per weight{WEIGHT_FORMATS[sel].extra ? ` (${WEIGHT_FORMATS[sel].extra})` : ''}, {fmt(cur.bytes / 1e9)} GB
        of weights read per token, {fmt(cur.ai)} FLOP/B, at {fmt(hw.gbs)} GB/s.
      </p>
      <table className="data ceiling-table" style={{ margin: 0 }}>
        <thead>
          <tr>
            <th>Format</th>
            <th className="num">Bits per weight</th>
            <th className="num">Weights, GB</th>
            <th className="num">FLOP/B</th>
            <th className="num">Ceiling, tokens/s</th>
            <th className="num">× BF16</th>
            <th style={{ width: '22%' }}></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.f} data-format={r.f} style={r.f === sel ? { background: 'var(--rule-soft)' } : undefined}>
              <td>{WEIGHT_FORMATS[r.f].label}</td>
              <td className="num">{fmt(r.bits)}</td>
              <td className="num">{fmt(r.bytes / 1e9)}</td>
              <td className="num">{fmt(r.ai)}</td>
              <td className="num">{fmt(r.ceil)}</td>
              <td className="num">{fmt(r.ceil / bf16.ceil)}</td>
              <td>
                <span style={{ display: 'block', height: '0.7rem', width: `${(100 * r.ceil) / top}%`, background: 'var(--global)', borderRadius: 2 }} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- near the top of E4M3
const TOP_CASES: { x: number; f: Format }[] = [
  { x: 448, f: E4M3 },
  { x: 460, f: E4M3 },
  { x: 464, f: E4M3 },
  { x: 470, f: E4M3 },
  { x: 57344, f: E5M2 },
  { x: 61440, f: E5M2 },
];

const big = (v: number) => (Number.isFinite(v) ? v.toLocaleString('en-US') : show(v));

function TopReveal({ v }: { v: number }) {
  return (
    <>
      <p>
        <b>448</b>, and here non-saturating mode gives 448 too. Near the top of E4M3 the values are 32 apart (exponent 8,
        three mantissa bits: 2<sup>8−3</sup>), so 460 sits between 448 and a would-be 480 and rounds down before any
        overflow rule applies.{' '}
        {v === 448 ? 'That is what you had.' : v === 460 ? '460 itself is not an E4M3 value.' : v > 448 ? 'Nothing above 448 exists in E4M3.' : 'Round to the nearest value; 448 is the closest.'}
      </p>
      <table className="data" style={{ margin: '0.4rem 0' }} data-reveal="e4m3">
        <thead>
          <tr>
            <th>Input</th>
            <th>To</th>
            <th>Saturating</th>
            <th>Non-saturating</th>
          </tr>
        </thead>
        <tbody>
          {TOP_CASES.map(({ x, f }) => {
            const s = convert(x, f, 'saturate');
            const n = convert(x, f, 'ieee');
            return (
              <tr key={x} data-input={x}>
                <td className="num">{x.toLocaleString('en-US')}</td>
                <td>{f.name.replace('FP8 ', '')}</td>
                <td data-sat={s.value}>{big(s.value)} <span className="muted mono">{showBits(f, s.code)}</span></td>
                <td data-nosat={String(n.value)} className={Number.isFinite(n.value) ? '' : 't-fail'}>
                  {big(n.value)} <span className="muted mono">{n.code >= 0 ? showBits(f, n.code) : ''}</span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p>
        Past 464, the halfway point, 470 rounds up to 480, which E4M3 does not have: saturating mode clamps to 448,
        non-saturating mode gives NaN because E4M3 has no infinity. E5M2 behaves like IEEE: 61,440 is halfway between
        its largest value, 57,344, and the next power of two, the tie rounds up, and it overflows to infinity.
      </p>
    </>
  );
}

// ---------------------------------------------------------------- one MX block
const OUTLIER_AT = 7;
const ELEM_LABEL: Record<ElemId, string> = { E2M1: 'FP4 E2M1', E2M3: 'FP6 E2M3', E3M2: 'FP6 E3M2', E4M3: 'FP8 E4M3', E5M2: 'FP8 E5M2' };

function MxBlockBench() {
  const [seed, setSeed] = useState(1);
  const [outlier, setOutlier] = useState(0);
  const [elem, setElem] = useState<ElemId>('E2M1');
  const [k, setK] = useState(32);
  const [rule, setRule] = useState<ScaleRule>('ocp-floor');
  const values = useMemo(() => sampleBlock(seed, 32, 0.25, outlier, OUTLIER_AT), [seed, outlier]);
  const q = useMemo(() => quantizeBlock(values, { elem, k, scale: rule }), [values, elem, k, rule]);
  const f = ELEMENTS[elem];
  const named = namedFormat({ elem, k, scale: rule });
  const nonzero = values.filter((v) => v !== 0).length;

  // Log2 magnitude axis, with a lane for zero underneath.
  const W = 720;
  const H = 300;
  const CX0 = 64;
  const cw = 20;
  const L0 = -12;
  const L1 = 7;
  const yTop = 18;
  const yBot = 236;
  const zeroY = 262;
  const ly = (a: number) => {
    if (a === 0) return zeroY;
    const l = Math.max(L0, Math.min(L1, Math.log2(a)));
    return yBot - ((l - L0) / (L1 - L0)) * (yBot - yTop);
  };
  const smallGrid = elemGrid(f).length <= 32;
  return (
    <Bench
      title="One MX block"
      task="Inject an outlier, then try to win back the small weights: smaller blocks, another scale rule, wider elements."
      foot={
        <>
          Magnitudes on a log scale; signs are stored exactly and not drawn. The grey band is what the block can represent
          (its scale times the element format's smallest and largest values); anything below the dashed line rounds to 0.
          32 weights drawn from a normal distribution, σ = 0.25.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Outlier at element {OUTLIER_AT}: {outlier === 0 ? 'none' : outlier}
          <input type="range" min="0" max="100" value={outlier} aria-label="outlier" onChange={(e) => setOutlier(+e.target.value)} />
        </label>
        <button onClick={() => setSeed((x) => x + 1)}>New weights</button>
        <label className="field">
          Element format
          <select value={elem} onChange={(e) => setElem(e.target.value as ElemId)} aria-label="element format">
            {ELEMENT_IDS.map((id) => (
              <option key={id} value={id}>{ELEM_LABEL[id]}</option>
            ))}
          </select>
        </label>
        <label className="field">
          Elements per scale
          <span className="seg">
            {[32, 16].map((n) => (
              <button key={n} aria-pressed={k === n} onClick={() => setK(n)}>{n}</button>
            ))}
          </span>
        </label>
        <label className="field">
          Scale rule
          <select value={rule} onChange={(e) => setRule(e.target.value as ScaleRule)} aria-label="scale rule">
            {(Object.keys(SCALE_RULES) as ScaleRule[]).map((r) => (
              <option key={r} value={r}>{SCALE_RULES[r]}</option>
            ))}
          </select>
        </label>
      </div>
      {/* On a phone the drawing keeps a readable width and scrolls sideways in its own frame. */}
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Weights and their decoded values" style={{ width: '100%', minWidth: 560, maxWidth: W, display: 'block' }}
          data-zeroed={q.zeroed} data-clamped={q.clamped} data-k={k} data-elem={elem} data-rule={rule} data-seed={seed} data-outlier={outlier}
          data-scales={q.blocks.map((b) => b.scale).join(',')}>
          {[-12, -8, -4, 0, 4].map((e) => (
            <g key={e}>
              <line x1={CX0 - 6} x2={CX0 + 32 * cw} y1={ly(2 ** e)} y2={ly(2 ** e)} stroke="var(--rule-soft)" />
              <text x={CX0 - 10} y={ly(2 ** e) + 4} fontSize="11" textAnchor="end">{e === 0 ? '1' : `2^${e}`}</text>
            </g>
          ))}
          <text x={CX0 - 10} y={zeroY + 4} fontSize="11" textAnchor="end">0</text>
          <line x1={CX0 - 6} x2={CX0 + 32 * cw} y1={zeroY} y2={zeroY} stroke="var(--rule)" />
          {q.blocks.map((b, j) => {
            const x0 = CX0 + b.start * cw;
            const x1 = x0 + b.size * cw;
            const hi = ly(b.scale * maxFinite(f));
            const lo = ly(b.scale * minSubnormal(f)!);
            const half = ly((b.scale * minSubnormal(f)!) / 2);
            return (
              <g key={j}>
                <rect x={x0} y={hi} width={x1 - x0} height={Math.max(0, lo - hi)} fill="var(--rule-soft)" />
                {smallGrid
                  ? elemGrid(f).slice(1).map((g) => (
                      <line key={g} x1={x0} x2={x1} y1={ly(b.scale * g)} y2={ly(b.scale * g)} stroke="var(--rule)" strokeWidth="0.7" />
                    ))
                  : null}
                <line x1={x0} x2={x1} y1={half} y2={half} stroke="var(--fail)" strokeDasharray="4 3" />
                {j > 0 ? <line x1={x0} x2={x0} y1={yTop - 8} y2={zeroY + 10} stroke="var(--ink)" strokeWidth="2" /> : null}
                <text x={x0 + 4} y={yTop - 4} fontSize="11">X = {b.scaleExp != null ? `2^${b.scaleExp}` : show(b.scale, 4)}</text>
              </g>
            );
          })}
          {values.map((v, i) => {
            const x = CX0 + i * cw + cw / 2;
            const d = q.decoded[i];
            const lost = v !== 0 && d === 0;
            const clamped = Math.abs(q.scaled[i]) > maxFinite(f);
            const bad = lost || clamped;
            return (
              <g key={i} data-i={i} data-lost={lost ? 1 : 0}>
                <line x1={x} x2={x} y1={ly(Math.abs(v))} y2={ly(Math.abs(d))} stroke={bad ? 'var(--fail)' : 'var(--ink-3)'} strokeWidth="1.6" />
                <circle cx={x} cy={ly(Math.abs(v))} r="4" fill="none" stroke="var(--global)" strokeWidth="1.3" />
                <circle cx={x} cy={ly(Math.abs(d))} r="3" fill={bad ? 'var(--fail)' : 'var(--global)'} />
              </g>
            );
          })}
          <text x={CX0} y={H - 6} fontSize="11" style={{ fill: 'var(--ink-2)' }}>element 0 … 31: open circle = weight, dot = what decodes back</text>
        </svg>
      </div>
      <div className="cols" style={{ marginTop: '0.8rem' }}>
        <div className="panel">
          <p style={{ margin: '0 0 0.5rem' }}>
            <b>{named ?? 'Not a documented format'}</b>
            <span className="muted"> · {fmt(q.bitsPerWeight)} bits per weight</span>
          </p>
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th>Block</th>
                <th className="num">max |v|</th>
                <th>Scale X</th>
                <th className="num">Byte</th>
                <th className="num">To 0</th>
                <th className="num">Clamped</th>
              </tr>
            </thead>
            <tbody>
              {q.blocks.map((b, j) => (
                <tr key={j} data-block={j}>
                  <td>{j + 1}</td>
                  <td className="num">{show(b.amax, 3)}</td>
                  <td>{b.scaleExp != null ? <>2<sup>{b.scaleExp}</sup> = {show(b.scale, 4)}</> : show(b.scale, 4)}</td>
                  <td className="num mono">{b.scaleCode.toString(16).padStart(2, '0')}</td>
                  <td className={'num' + (b.zeroed ? ' t-fail' : '')}>{b.zeroed}</td>
                  <td className={'num' + (b.clamped ? ' t-fail' : '')}>{b.clamped}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <p style={{ margin: 0 }} data-summary>
            <span className={q.zeroed ? 'chip fail' : 'chip ok'}>{q.zeroed} of {nonzero} lost to 0</span>{' '}
            {q.clamped ? <span className="chip fail">{q.clamped} clamped</span> : null}
          </p>
          <p className="muted" style={{ margin: '0.5rem 0 0' }}>
            RMS error {show(100 * q.rmsRel, 3)}% of the block's RMS; {show(100 * q.rmsRelRest, 3)}% for the 31 weights other than the
            largest.
          </p>
        </div>
      </div>
      <details style={{ marginTop: '0.8rem' }}>
        <summary>All 32 elements</summary>
        <table className="data" style={{ margin: '0.4rem 0 0' }} data-elements>
          <thead>
            <tr>
              <th className="num">i</th>
              <th className="num">weight v</th>
              <th className="num">v / X</th>
              <th>stored code</th>
              <th className="num">decodes to</th>
              <th className="num">error</th>
            </tr>
          </thead>
          <tbody>
            {values.map((v, i) => (
              <tr key={i}>
                <td className="num">{i}</td>
                <td className="num">{show(v, 4)}</td>
                <td className="num">{show(q.scaled[i], 4)}</td>
                <td className="mono">{showBits(f, q.codes[i])}</td>
                <td className={'num' + (v !== 0 && q.decoded[i] === 0 ? ' t-fail' : '')}>{show(q.decoded[i], 4)}</td>
                <td className="num">{show(q.err[i], 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSA5: Record<string, Question> = {
  'appA/A.5/q-h100': {
    kind: 'choice',
    id: 'appA/A.5/q-h100',
    prompt: (
      <>
        The book sets the H100's 3,958 TFLOPS for FP8 on tensor cores against 989 TFLOPS it calls single precision on
        tensor cores. What is the 989, by NVIDIA's own table?
      </>
    ),
    options: [
      { label: 'IEEE FP32 matrix throughput on tensor cores, dense.', why: "NVIDIA's row is TF32, and it carries the 'with sparsity' footnote." },
      { label: 'TF32 tensor-core throughput with sparsity.', correct: true },
      { label: 'FP32 throughput of the ordinary cores, no tensor cores.', why: 'NVIDIA lists that as 67 TFLOPS; 989 is a tensor-core row.' },
      { label: 'FP16 tensor-core throughput with sparsity.', why: 'FP16 and BF16 are the 1,979 row.' },
    ],
    explain: (
      <p>
        Both of the book's numbers are NVIDIA's "with sparsity" rows, and 989 is TF32: FP32's exponent range with at
        least 10 mantissa bits, a tensor-core input format rather than IEEE single precision. The four-to-one ratio holds
        between the two sparse rows, so the book's point survives: fewer bits per value, more operations per second.
      </p>
    ),
    source: 'book §A.5, p. 593; notes/today/appA-facts.md A11 and ch05-facts.md A8',
  },
  'appA/A.5/q-ceiling': {
    kind: 'numeric',
    id: 'appA/A.5/q-ceiling',
    prompt: <>A model's weights move from BF16 to MXFP4. By what factor does its batch-1 decode ceiling rise, by the bandwidth argument above?</>,
    answer: 3.76,
    tol: 0.02,
    unit: '×',
    explain: (
      <p>
        The ceiling is bandwidth divided by bytes per token, so it scales with 1 / bits per weight: 16 / 4.25 = 3.76. Not
        4: MXFP4's scales cost a quarter of a bit per weight. A real decode step also reads the KV cache and activations,
        which do not shrink with the weights, so the measured gain is smaller.
      </p>
    ),
    source: 'ch. 5 §5.1 method; MX v1.1 Table 1 (notes/today/appA-facts.md A4); our arithmetic',
  },
  'appA/A.5/q-outlier': {
    kind: 'choice',
    id: 'appA/A.5/q-outlier',
    prompt: <>An MXFP4 block holds one weight of 100 and 31 weights near 0.3. Under the OCP scale rule, what do the 0.3s decode to?</>,
    options: [
      { label: '0: divided by the scale, 0.3 is under half the smallest step.', correct: true },
      { label: 'About 0.3, since each element keeps an exponent of its own.', why: 'An element has 2 exponent bits; the range comes from the one shared scale.' },
      { label: '8: E2M1\'s smallest step, 0.5, times the block scale of 16.', why: '0.3 / 16 ≈ 0.019 is far closer to 0 than to 0.5.' },
      { label: '96: they clamp to the largest value times the scale.', why: 'Only elements above the largest value clamp; these are tiny.' },
    ],
    explain: (
      <p>
        floor(log2 100) = 6; minus E2M1's largest exponent, 2, gives a scale of 2<sup>4</sup> = 16. Then 100 / 16 = 6.25
        clamps to 6 (decoding to 96), and 0.3 / 16 ≈ 0.019 is below 0.25, half of E2M1's smallest step, so it rounds to
        0. One outlier erased 31 weights. Blocks of 16, or finer scales, confine it: try both in the explorer.
      </p>
    ),
    source: 'MX v1.1 §6.3 rule (notes/today/appA-facts.md A4); our arithmetic',
  },
  'appA/A.5/q-mxfp4-bits': {
    kind: 'numeric',
    id: 'appA/A.5/q-mxfp4-bits',
    prompt: <>How many bits per weight does MXFP4 take, counting its scale?</>,
    answer: 4.25,
    tol: 0.001,
    unit: 'bits',
    explain: (
      <p>
        32 elements of 4 bits plus one 8-bit UE8M0 scale: 136 bits for 32 weights, 4.25 per weight. llama.cpp's{' '}
        <code>block_mxfp4</code> is exactly that: 17 bytes per 32 weights.
      </p>
    ),
    source: 'MX v1.1 Table 1; llama.cpp ggml-common.h (notes/today/appA-facts.md A4, D3)',
  },
  'appA/A.5/q-fnuz': {
    kind: 'choice',
    id: 'appA/A.5/q-fnuz',
    prompt: <>OCP E4M3 bytes are handed, unconverted, to code that reads them as CDNA3's FNUZ E4M3. What happens?</>,
    options: [
      { label: 'Finite values halve, 0x80 turns into NaN, and the OCP NaN byte becomes ±240.', correct: true },
      { label: 'Nothing changes: both formats have 1 sign, 4 exponent and 3 mantissa bits.', why: 'Same field widths, different bias and different special values.' },
      { label: 'Finite values double, because the two exponent biases differ by one.', why: 'FNUZ has bias 8, one more than OCP: the same bits mean a smaller value.' },
      { label: 'Only values above 240 change: they saturate to 240, the FNUZ maximum.', why: 'The bias differs, so every non-zero value changes.' },
    ],
    explain: (
      <p>
        FNUZ E4M3 has bias 8 where OCP has 7, so the same exponent bits stand for one power of two less: every finite
        value halves. FNUZ has no negative zero and uses 1000 0000 as its NaN, while OCP's NaN pattern S.1111.111 is an
        ordinary number in FNUZ, 1.875 × 2<sup>7</sup> = 240. That is why HIP has separate <code>__hip_fp8_e4m3</code> and{' '}
        <code>__hip_fp8_e4m3_fnuz</code> types and PyTorch separate <code>float8_e4m3fn</code> and <code>float8_e4m3fnuz</code>.
      </p>
    ),
    source: 'notes/today/appA-facts.md B2 (bias from HIP\'s header), D1; our arithmetic',
  },
};

// ---------------------------------------------------------------- the section
const FORMAT_ROWS: { f: Format; label: string; sem: string; bias: number; inf: string; nan: string }[] = [
  { f: E4M3, label: 'FP8 E4M3', sem: '1-4-3', bias: 7, inf: 'none', nan: 'S.1111.111' },
  { f: E5M2, label: 'FP8 E5M2', sem: '1-5-2', bias: 15, inf: 'S.11111.00', nan: 'S.11111.{01, 10, 11}' },
  { f: E3M2, label: 'FP6 E3M2', sem: '1-3-2', bias: 3, inf: 'none', nan: 'none' },
  { f: E2M3, label: 'FP6 E2M3', sem: '1-2-3', bias: 1, inf: 'none', nan: 'none' },
  { f: E2M1, label: 'FP4 E2M1', sem: '1-2-1', bias: 1, inf: 'none', nan: 'none' },
];

export function SA5() {
  const mx = useLabFile('appA/snippets/mx_quantize.h');
  return (
    <>
      <p>
        The book gives lower-precision formats two paragraphs. The first says why they exist. Some applications work on
        small data types by nature, 4, 6, 8 or 16 bits wide: the book names medical imaging, remote sensing, radio
        astronomy and seismic analysis, and LLMs have more recently been adapted to small floating-point types too. Such data needs only a few
        significant bits, but it still benefits from range, which is what exponent bits buy (§A.2). That is the case for
        floating-point formats narrower than single precision rather than small integers.
      </p>
      <Source section="A.5" pages="592–593" />
      <p>
        The second paragraph is about speed. A smaller format moves fewer bytes for each arithmetic operation, which
        raises the compute-to-memory ratio chapter 5 used to place a kernel on the roofline, and GPUs now compute in
        16-, 8- and 4-bit floating point directly. The book's example is the H100: 3,958 TFLOPS for FP8 on its tensor
        cores against 989 TFLOPS for single precision, four times as many operations per second.
      </p>
      <Today checked="2026-11-20 (notes/today/appA-facts.md A10, A11; ch05-facts.md A8)">
        <p>
          Both figures are NVIDIA's "with sparsity" rows, and the 989 is TF32 (FP32's range, at least 10 mantissa bits),
          not IEEE single precision. FP8 tensor cores start at compute capability 8.9; FP6 and FP4 at 10.0.
        </p>
      </Today>
      <Ask q={questionsSA5['appA/A.5/q-h100']} />

      <h2>What fewer bits buy a decode step</h2>
      <p>
        Chapter 5 counted the decode step at batch 1: every weight is read once and used for one multiply and one add.
        Nothing reuses a weight, so the step is bound by memory, and its ceiling is the bandwidth divided by the bytes of
        weights read per token. Halve the bits per weight and the ceiling doubles, though no arithmetic got faster. That
        is the bandwidth argument for quantised weights.
      </p>
      <Beyond>
        <p>
          The book's §A.5 stops at "fewer bytes per operation". This is chapter 5's method (<a href={href('ch05', '5.1')}>§5.1</a>)
          applied to the formats below; the block formats themselves come from OCP and NVIDIA.
        </p>
      </Beyond>
      <DecodeCeilingBench />
      <p>
        Read the table down. Every row's ceiling is the BF16 row's times 16 over its bits per weight, whatever the model
        size or the device. Block formats pay for their scales: one 8-bit scale per 32 weights adds a quarter of a bit,
        one per 16 adds half a bit. Select your own device if you have run lab 5.1, and the whole table moves together.
      </p>
      <Simplification>
        <p>
          Weights only: no KV cache, activations or launch overhead; batch 1; the full bandwidth used; per-tensor scales
          ignored. Real decode reaches a fraction of these ceilings, and the KV cache does not shrink with the weights.
        </p>
      </Simplification>
      <Ask q={questionsSA5['appA/A.5/q-ceiling']} />

      <h2>The layouts the book leaves out</h2>
      <p>
        The book names no bit layout for anything below 16 bits. The definitions that hardware and frameworks now share
        come from the Open Compute Project: the OFP8 specification for the two 8-bit formats, and the MX specification
        for the 6- and 4-bit ones and for the scale type. Everything in §A.1 to §A.3 applies to them, with one twist: the
        narrow formats give up infinities, and sometimes NaN, to keep more finite values.
      </p>
      <Beyond>
        <p>
          From OCP OFP8 r1.1 and MX v1.1 (notes/today/appA-facts.md A3, A4). The book cites MX v1.0 (2023); v1.1 renamed
          the scale type UE8M0.
        </p>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-formats>
        <thead>
          <tr>
            <th>Format</th>
            <th>S-E-M bits</th>
            <th className="num">Bias</th>
            <th className="num">Largest</th>
            <th className="num">Smallest positive</th>
            <th>Infinity</th>
            <th>NaN</th>
          </tr>
        </thead>
        <tbody>
          {FORMAT_ROWS.map((r) => (
            <tr key={r.label}>
              <td>{r.label}</td>
              <td className="mono">{r.sem}</td>
              <td className="num">{r.bias}</td>
              <td className="num">{maxFinite(r.f).toLocaleString('en-US')}</td>
              <td className="num">
                2<sup>{Math.log2(minSubnormal(r.f)!)}</sup>
              </td>
              <td className="mono">{r.inf}</td>
              <td className="mono">{r.nan}</td>
            </tr>
          ))}
          <tr>
            <td>UE8M0 (scales)</td>
            <td className="mono">0-8-0</td>
            <td className="num">127</td>
            <td className="num">
              2<sup>127</sup>
            </td>
            <td className="num">
              2<sup>−127</sup>
            </td>
            <td className="mono">none</td>
            <td className="mono">1111 1111; no zero either</td>
          </tr>
        </tbody>
      </table>
      <p>
        E2M1 has just eight magnitudes: {elemGrid(E2M1).join(', ')}. On its own that is useless for weights. It becomes
        useful when a block of elements shares a scale, which is the subject of the next part of this page. E5M2 keeps
        IEEE's special values and is FP16 with eight mantissa bits cut off; E4M3 spends its top exponent on numbers,
        keeping a single NaN pattern, and reaches 448 where an IEEE-style layout would stop at 240.
      </p>

      <h3>Converting into them</h3>
      <p>
        Conversion rounds to nearest, ties to even, the rule of §A.4. What happens past the largest value is a separate
        choice: saturating mode clamps to it; non-saturating mode gives NaN for E4M3, which has no infinity, and infinity
        for E5M2. Rounding happens first, so a value a little past the top may round down to it before either rule
        applies.
      </p>
      <Today checked="2026-11-20 (notes/today/appA-facts.md A3, A5, B2, C4)">
        <p>
          Defaults differ. CUDA's <code>__nv_fp8_e4m3(float)</code> and HIP's <code>__hip_fp8_e4m3</code> saturate. Metal's{' '}
          <code>pack</code> saturates for E4M3 and E2M1 but not for E5M2. The 464 and 61,440 cases are in OFP8 r1.1's new
          Appendix A.
        </p>
      </Today>
      <Predict id="appA/A.5/p-e4m3" prompt={<>460 is converted to E4M3 in saturating mode. What value is stored?</>}>
        {(v) => <TopReveal v={v} />}
      </Predict>

      <h2>One scale for a block of weights</h2>
      <p>
        Block scaling stores a group of k elements P<sub>i</sub> and one shared scale X, and the value of element i is X
        · P<sub>i</sub>. In OCP's MX formats k is 32 and X is a power of two, stored as one UE8M0 byte: MXFP8 has E4M3 or
        E5M2 elements, MXFP6 E3M2 or E2M3, MXFP4 E2M1. The scale gives the block its range; the elements give it a few
        significant bits.
      </p>
      <Beyond>
        <p>From MX v1.1 §§5–6 (notes/today/appA-facts.md A4). Integer block formats (MXINT8) exist too and are left out here.</p>
      </Beyond>
      <p>
        OCP suggests choosing X from the largest magnitude in the block: the shared exponent is floor(log2 max|V|) minus
        the largest exponent of the element format, and any element that then lands above the element format's largest
        value is clamped to it. cuBLAS rounds the exponent up instead, which, in its own words, "differs from the OCP
        MXFP suggested rounding scheme": the largest element then never clamps, at the cost of one binade for the rest
        some of the time.
      </p>
      <Today checked="2026-11-20 (notes/today/appA-facts.md A5, D2, D3)">
        <p>
          NVIDIA's FP4 GEMMs in cuBLAS use another layout: 16 elements per scale, the scale an 8-bit float (UE4M3) instead
          of a power of two, plus one scale per tensor. NVIDIA's pages do not call it NVFP4; MLX and llama.cpp do.
        </p>
      </Today>
      <MxBlockBench />
      <p>
        Start with no outlier and E2M1: a few of the smallest weights already round to 0. Now raise the outlier. Its
        magnitude sets the scale for all 32, and the small weights fall below the dashed line one after another. Then
        win them back three ways: 16 elements per scale, so the outlier only spoils its own half; E4M3 elements, whose
        extra exponent bits reach far below the scale; or the E4M3 scale rule, which fits the scale to the block more
        finely than a power of two. Compare the OCP and cuBLAS rules for which one clamps.
      </p>
      <Simplification>
        <p>
          One block in isolation, normally distributed weights. The E4M3 scale is modelled as a positive E4M3 value with
          the tensor-wide scale set to 1 (UE4M3's exact encoding is not in the sources we checked). An all-zero block takes
          the smallest scale, our choice; it decodes to zeros either way.
        </p>
      </Simplification>
      <Ask q={questionsSA5['appA/A.5/q-outlier']} />
      <Ask q={questionsSA5['appA/A.5/q-mxfp4-bits']} />

      <h3>Quantising one block, step by step</h3>
      <p>
        The pipeline is short, and one early decision ripples through it: find the largest magnitude, derive the scale,
        divide, round onto the grid, clamp, store. Watch it once on a block with an outlier.
      </p>
      <MxBlockScene />
      <p>The same pipeline in host C++, the shape a CPU reference for a quantisation kernel takes:</p>
      <Code
        tabs={[{ label: 'C++ (host)', code: mx ?? '// loading labs/appA/snippets/mx_quantize.h' }]}
        caption={
          <>
            labs/appA/snippets/mx_quantize.h, ours. The unit tests compile it, run it on the scene's block and fourteen
            others, and compare every scale byte, code and decoded float with the model behind the explorer, bit for bit.
          </>
        }
      />
      <p>
        One more property of the formats connects back to this appendix. MX v1.1 states that "the internal precision of
        the dot product and order of operations is implementation-defined". Two conforming kernels may sum the 32
        products of a block in different orders and agree only to within rounding: §
        <a href={href('appA', 'A.6')}>A.6</a>'s point, at the scale of a block.
      </p>

      <h2>Who has what, today</h2>
      <p>
        None of this is in the book; all of it is in current vendor documentation. The short version: CUDA and HIP have
        scalar types for the small formats that you convert to and from, tensor cores and matrix units compute with them,
        and Metal has them only as packed storage and tensor element types.
      </p>
      <Today checked="2026-11-20 (notes/today/appA-facts.md A5, A10, B2, C4, D1 to D3)">
        <p>
          CUDA's small types support queries such as <code>numeric_limits</code>; NVIDIA lists no arithmetic operators
          for them. Apple's Tables 2.15 and 2.16 contain three misprints (use OCP's tables), and Apple does not say which
          GPU families support these formats.
        </p>
      </Today>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr>
            <th></th>
            <th>FP8</th>
            <th>FP6, FP4</th>
            <th>Block scales</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>CUDA</td>
            <td>
              <code>__nv_fp8_e4m3</code>, <code>__nv_fp8_e5m2</code> (<code>cuda_fp8.h</code>); tensor cores from compute
              capability 8.9
            </td>
            <td>
              <code>__nv_fp6_e2m3</code>, <code>__nv_fp6_e3m2</code>, <code>__nv_fp4_e2m1</code> (<code>cuda_fp6.h</code>,{' '}
              <code>cuda_fp4.h</code>); tensor cores from 10.0
            </td>
            <td>
              <code>__nv_fp8_e8m0</code>; PTX <code>mma</code> with block scaling; cuBLAS 32-element UE8M0 for FP8, 16-element
              UE4M3 for FP4
            </td>
          </tr>
          <tr>
            <td>HIP</td>
            <td>
              <code>__hip_fp8_e4m3</code>, <code>_e5m2</code> (OCP; CDNA4, RDNA4); <code>_fnuz</code> (CDNA3: bias one
              higher, no infinity, no −0, NaN 0x80)
            </td>
            <td>CDNA4 only</td>
            <td>
              <code>__amd_scale_t</code> (E8M0) conversion APIs, hardware on gfx950
            </td>
          </tr>
          <tr>
            <td>Metal 4.1</td>
            <td>
              <code>metal_fp8_e4m3_format</code>, <code>metal_fp8_e5m2_format</code>: packed storage and tensor element types;
              no scalar FP8 arithmetic
            </td>
            <td>
              <code>metal_fp4_e2m1_format</code>; no FP6
            </td>
            <td>
              <code>tensor_blockwise</code> with <code>metal_fp8_ue8m0_format</code>, block 32
            </td>
          </tr>
          <tr>
            <td>Frameworks</td>
            <td>
              PyTorch's "shell" dtypes such as <code>float8_e4m3fn</code> ("fn": finite only; "uz": unsigned zero)
            </td>
            <td>
              MLX <code>quantize</code> modes <code>mxfp4</code>, <code>mxfp8</code>, <code>nvfp4</code>
            </td>
            <td>
              llama.cpp <code>block_mxfp4</code>: 17 bytes per 32 weights, its E2M1 table stored doubled to fit{' '}
              <code>int8_t</code>
            </td>
          </tr>
        </tbody>
      </table>
      <p>
        The FNUZ row is the trap for anyone reading AMD and NVIDIA code side by side: same field widths, different
        numbers.
      </p>
      <Ask q={questionsSA5['appA/A.5/q-fnuz']} />

      <h2>Where this goes next</h2>
      <p>
        Integer quantisation, int8 or 4-bit codes with a scale and a zero point, is not in the book at all, and neither is
        a kernel that computes with any of these formats. Module <a href={href('x2')}>X2</a> builds both on this page: a
        quantised GEMM whose inner loop decodes blocks like the one above, and milestone M9, which measures the decode
        speed-up the first explorer predicts.
      </p>
    </>
  );
}
