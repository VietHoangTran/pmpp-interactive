import { Fragment, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { FORMATS, bitsPerWeight, codeBitsPerWeight, layer, gemv, modelBytes, type FormatId } from '../../sim/qtraffic';

// Module X2's start page: what this module is (ours, anchored on appendix A §A.5 and
// chapter 5's intensity), which kernel a quantised linear layer ends in today (a small
// classification table, data only), the size of the problem (a Predict whose reveal is
// computed by qtraffic.ts), the plan, and six prerequisite questions from earlier chapters.

const FACTS = 'notes/today/x2-facts.md';
const PARAMS = 7e9;
const gb = (b: number) => (b / 1e9).toFixed(2);

export const questionsSX20: Record<string, Question> = {
  'x2/start/q-gemv-intensity': {
    kind: 'choice',
    id: 'x2/start/q-gemv-intensity',
    prompt: (
      <>
        The book's linear layer, 1024 inputs to 4096 outputs, does about 1 FLOP per byte at batch 1 in 16-bit (
        <a href={href('ch05', '5.1')}>section 5.1</a>, and chapter 20, p. 505). What is it with float32 weights and
        activations, and what could raise it?
      </>
    ),
    options: [
      {
        label: 'About 0.5 FLOP/B; only batching or fewer bytes per weight raise it.',
        correct: true,
      },
      { label: 'About 0.5 FLOP/B; a large enough shared-memory tile raises it.', why: 'Tiling removes repeated fetches. At batch 1 no weight is fetched twice: there is nothing to share.' },
      { label: 'About 2 FLOP/B: twice the bytes means twice the work per byte.', why: 'The work is the same; the bytes double, so the ratio halves.' },
      { label: 'About 1 FLOP/B: the number format does not change the ratio.', why: 'The ratio is operations per byte, and float32 moves twice the bytes for the same operations.' },
    ],
    explain: (
      <p>
        2 × 1024 × 4096 operations over about 4 × 1024 × 4096 bytes of weights is 0.5 FLOP/B, plus a sliver for the
        input and output vectors. Every weight takes part in one multiply and one add per token, so only more rows per
        weight (batching) or fewer bytes per weight move the ratio. That is why this module
        exists: the format is one of the two knobs left.
      </p>
    ),
    source: 'ch. 5 §5.1, pp. 93–97; ch. 20 §20.6, p. 505',
  },
  'x2/start/q-gather': {
    kind: 'choice',
    id: 'x2/start/q-gather',
    prompt: (
      <>
        In lab 22.1's gather GEMV (<a href={href('ch22', 'labs')}>lab 22.1</a>), which thread writes Y[b][j], and why does
        it need no atomic?
      </>
    ),
    options: [
      { label: 'Lane 0 of the one warp that owns output j, after its shuffle reduction.', correct: true },
      { label: 'Every lane of the warp, each with its partial sum, so the write is atomic.', why: 'That is split-K, the scatter version of the same lab. The gather version reduces first.' },
      { label: 'Thread j of the grid, which loops over the whole row on its own.', why: 'That is one thread per output. Lab 22.1 gives each output a warp so that the row loads coalesce.' },
      { label: 'The last warp of the block, after a barrier collects the partials.', why: 'There is no cross-warp step: one warp owns one output.' },
    ],
    explain: (
      <p>
        One warp per output row is a gather: the warp's lanes read the row in step, a shuffle reduction collects their
        partial sums in lane 0, and lane 0 writes. Section X2.4 keeps this shape and changes only what a lane loads.
      </p>
    ),
    source: 'ch. 22 §22.2, pp. 532–533; lab 22.1 README',
  },
  'x2/start/q-coalesce': {
    kind: 'choice',
    id: 'x2/start/q-coalesce',
    prompt: (
      <>
        W is stored (out, in) row-major and lane L of the warp for output j reads W[j·in + k0 + L]. Are those loads
        coalesced? And with W stored (in, out), lane L reading W[(k0 + L)·out + j]? (<a href={href('ch06', '6.1')}>section 6.1</a>)
      </>
    ),
    options: [
      { label: 'Yes for (out, in); no for (in, out), where lanes are out floats apart.', correct: true },
      { label: 'Both are coalesced, because every lane reads exactly one float.', why: 'Coalescing is about the addresses of neighbouring lanes, not the size of each read.' },
      { label: 'Neither: each weight is read only once, so nothing can be combined.', why: 'Combining happens across the lanes of one load instruction, not across time.' },
      { label: 'No for (out, in), where lanes skip a row; yes for (in, out).', why: 'In (in, out) the lanes jump a whole row of out values each.' },
    ],
    explain: (
      <p>
        With (out, in), the 32 lanes of a warp touch 128 consecutive bytes, one run of segments. With (in, out) each lane
        is a row apart and every lane needs its own segment. llama.cpp and MLX both store weight rows as (out, in), and so
        does every lab in this module.
      </p>
    ),
    source: 'ch. 6 §6.1, pp. 124–132; x2-facts project notes',
  },
  'x2/start/q-fp16-int': {
    kind: 'numeric',
    id: 'x2/start/q-fp16-int',
    prompt: (
      <>
        FP16 has an 11-bit significand, counting the hidden bit (<a href={href('appA', 'A.2')}>section A.2</a>). Up to
        what n is every integer from 0 to n exactly representable in FP16?
      </>
    ),
    answer: 2048,
    explain: (
      <p>
        2¹¹ = 2048. Above that FP16 steps by 2, then by 4. Its largest finite value is 65,504. Keep both numbers: a block
        of int8 products sums to far more than either, which is the subject of section X2.6.
      </p>
    ),
    source: 'appA §A.2, pp. 586–590; notes/today/appA-facts.md A2; x2-facts E3',
  },
  'x2/start/q-acc-order': {
    kind: 'choice',
    id: 'x2/start/q-acc-order',
    prompt: (
      <>
        Why does adding 4096 products one at a time lose more in FP16 than in float32 (<a href={href('appA', 'A.6')}>section A.6</a>)?
      </>
    ),
    options: [
      {
        label: 'Small addends vanish against a large sum, and FP16 gets there far sooner.',
        correct: true,
      },
      { label: 'FP16 overflows after 2048 terms, and the rest of the sum is lost.', why: 'Overflow depends on the size of the sum, not the number of terms. The loss here happens long before any overflow.' },
      { label: 'FP16 rounds toward zero, while float32 rounds to the nearest value.', why: 'Both round to nearest by default. The difference is how many bits each keeps.' },
      { label: 'It does not: the order of the additions is all that matters.', why: 'Order matters in both. The narrower format makes the same order worse.' },
    ],
    explain: (
      <p>
        This is the book's disappearing operand: once the running sum is large, an addend smaller than half its
        last-place step is lost, and FP16 reaches that point after far fewer terms. A wider
        accumulator delays the point where it starts. Every quantised kernel in this module accumulates in int32 or
        float, never in the stored format.
      </p>
    ),
    source: 'appA §A.6, pp. 593–594',
  },
  'x2/start/q-warp-reduce': {
    kind: 'choice',
    id: 'x2/start/q-warp-reduce',
    prompt: (
      <>
        After a butterfly reduction with <code>__shfl_xor_sync</code>, which lanes hold the warp's total? And after one
        with <code>__shfl_down_sync</code> (<a href={href('ch10', '10.7')}>section 10.7</a>)?
      </>
    ),
    options: [
      { label: 'Every lane after the xor butterfly; only lane 0 after shuffle-down.', correct: true },
      { label: 'Only lane 0 in both cases, since lane 0 is where the sum ends.', why: 'In a butterfly each lane exchanges with a partner and both keep the sum, so all lanes end with it.' },
      { label: 'Every lane in both cases: each shuffle moves values to all lanes.', why: 'With shuffle-down the upper lanes receive values from past the end of the warp and their sums are partial.' },
      { label: 'The last lane after the butterfly; lane 0 after shuffle-down.', why: 'The butterfly is symmetric: no lane is special.' },
    ],
    explain: (
      <p>
        Lab 22.1 uses shuffle-down and lets lane 0 write. Metal's <code>simd_sum</code> gives the sum to every active lane.
        A GEMV that only needs one writer can use either.
      </p>
    ),
    source: 'ch. 10 §10.7, pp. 237–241; notes/today/ch10-facts.md (table "Which lanes hold the warp result")',
  },
};

// ---------------------------------------------------------------- what you run today
type Family = 'int4pack' | 'refused' | 'mlx-qmv' | 'mmvq' | 'mulmv';
const FAMILIES: { id: Family; label: string }[] = [
  { id: 'int4pack', label: "PyTorch's weight-only int4 core op, _weight_int4pack_mm" },
  { id: 'refused', label: 'refused: no kernel serves it' },
  { id: 'mlx-qmv', label: "MLX's quantised matrix-vector kernels (affine_qmv_fast and its siblings)" },
  { id: 'mmvq', label: 'mmvq: activations quantised to Q8_1, integer dot per block' },
  { id: 'mulmv', label: 'mul_mv: float multiply-adds on masked 4-bit codes' },
];
interface Call {
  id: string;
  call: string;
  answer: Family;
  why: string;
}
// Data only: what each call ends in, with the facts items that say so.
const CALLS: Call[] = [
  {
    id: 'torch-int4',
    call: 'PyTorch, int4 weights in groups of 128 with a BF16 scale and zero each, batch 1, on CUDA',
    answer: 'int4pack',
    why: "That is the rule of the core op _weight_int4pack_mm: its CUDA kernel dequantises each 4-bit value as q × scale + zero in BF16, and needs compute capability 8.0 or a ROCm build. Whether a torchao config such as Int4WeightOnlyConfig reaches this op depends on its packing format, which the facts pass did not check (D6).",
  },
  {
    id: 'int-mm',
    call: 'torch._int_mm with M = 1, on CUDA',
    answer: 'refused',
    why: 'The CUDA implementation checks that the first dimension is greater than 16 and stops otherwise. The int8 GEMM is not a decode kernel (A6).',
  },
  {
    id: 'mlx',
    call: 'mlx.nn.QuantizedLinear on a Mac',
    answer: 'mlx-qmv',
    why: "Defaults: group 64, 4 bits, affine. For one row its Metal kernels are the quantised matrix-vector family; affine_qmv_fast is the one read in the facts pass. Which sibling runs is chosen by shape, which the facts pass did not check (D5; infra-facts D3).",
  },
  {
    id: 'llama-cuda',
    call: 'llama.cpp, a Q4_0 tensor, on CUDA',
    answer: 'mmvq',
    why: 'Up to 8 tokens llama.cpp uses matrix-vector kernels. mmvq first quantises the float activations to Q8_1, then does integer dot products per block and one float rescale per block (ch15-facts D1; D3).',
  },
  {
    id: 'llama-metal',
    call: 'llama.cpp, a Q4_0 tensor, on Metal',
    answer: 'mulmv',
    why: 'Metal has no integer dot. llama.cpp masks the nibbles in place, divides the activations by 16, 256 or 4096 to make up for the missing shifts, and multiplies in float (D4; C1).',
  },
];

function KernelClassify() {
  const [pick, setPick] = useState<Record<string, Family | ''>>({});
  const [checked, setChecked] = useState(false);
  const right = CALLS.filter((c) => pick[c.id] === c.answer).length;
  const all = CALLS.every((c) => pick[c.id]);
  return (
    <Bench
      title="Which kernel serves this call?"
      task="For each call, pick the kernel family it ends in at decode time. Then check."
      foot={<>From the facts pass of 2026-11-20 ({FACTS}, items A6, D3 to D6). These projects change often; the item letters say where each answer was read.</>}
    >
      <div style={{ overflowX: 'auto' }}>
        <table className="data" data-table="classify" style={{ margin: 0, background: 'var(--page)' }}>
          <thead>
            <tr>
              <th>Call</th>
              <th>Ends in</th>
              {checked ? <th>Verdict</th> : null}
            </tr>
          </thead>
          <tbody>
            {CALLS.map((c) => {
              const ok = pick[c.id] === c.answer;
              return (
                <Fragment key={c.id}>
                  <tr data-call={c.id} data-verdict={checked ? (ok ? 'right' : 'wrong') : ''}>
                    <td style={{ minWidth: '11rem' }}>{c.call}</td>
                    <td>
                      <select
                        aria-label={`Kernel for ${c.call}`}
                        value={pick[c.id] ?? ''}
                        style={{ maxWidth: '100%' }}
                        onChange={(e) => {
                          setPick({ ...pick, [c.id]: e.target.value as Family });
                          setChecked(false);
                        }}
                      >
                        <option value="">choose</option>
                        {FAMILIES.map((f) => (
                          <option key={f.id} value={f.id}>{f.label}</option>
                        ))}
                      </select>
                    </td>
                    {checked ? <td>{ok ? <span className="chip ok">right</span> : <span className="chip fail">not this one</span>}</td> : null}
                  </tr>
                  {checked ? (
                    <tr data-why={c.id}>
                      <td colSpan={3} className="muted" style={{ whiteSpace: 'normal', paddingTop: 0 }}>
                        {ok ? '' : `It is ${FAMILIES.find((f) => f.id === c.answer)!.label}. `}
                        {c.why}
                      </td>
                    </tr>
                  ) : null}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        <button className="primary" disabled={!all} onClick={() => setChecked(true)}>Check</button>
        {checked ? <span data-score={right}>{right} of {CALLS.length} right.</span> : <span className="muted">Pick one for every call to check.</span>}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the size of the problem
const SIZE_FORMATS: FormatId[] = ['f32', 'f16', 'q8_0', 'q4_0'];

function SizeTable() {
  return (
    <table className="data" data-table="sizes" style={{ margin: '0.6rem 0 0' }}>
      <thead>
        <tr>
          <th>Format</th>
          <th className="num">Bits per weight</th>
          <th className="num">of which scales</th>
          <th className="num">7 billion weights, GB</th>
        </tr>
      </thead>
      <tbody>
        {SIZE_FORMATS.map((f) => (
          <tr key={f} data-format={f}>
            <td>{FORMATS[f].label}{f.startsWith('q') ? <span className="muted"> ({FORMATS[f].owner})</span> : null}</td>
            <td className="num" data-bits>{bitsPerWeight(f)}</td>
            <td className="num">{(bitsPerWeight(f) - codeBitsPerWeight(f)).toString()}</td>
            <td className="num" data-gb>{gb(modelBytes(PARAMS, f))}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function SX20() {
  const q = questionsSX20;
  const q4 = modelBytes(PARAMS, 'q4_0');
  const bookLayer = gemv('f16', layer('1024x4096'), 1, 2);
  return (
    <>
      <p>
        At batch 1, every linear layer of a decoder is a matrix-vector product, and every weight of the model crosses the
        memory bus once per generated token. You have met this twice: as the roofline of chapter 5 and as the decode step
        of chapters 20 and 22. This module turns the one knob those chapters left alone: how many bytes each weight takes.
      </p>
      <Beyond title="Ours, not the book's">
        <p>
          The book has no integer quantisation, scales, zero points or quantised kernels. X2 is built by us on its
          methods.
        </p>
      </Beyond>
      <p>
        The book's anchor is appendix A's §A.5, two paragraphs: a smaller format moves fewer bytes per operation, and so
        raises intensity. It stops there. Chapter 5 supplies the method that turns that sentence into a prediction you
        can measure, and chapters 10 and 22 the warp reductions and the gather GEMV this module builds on.
      </p>
      <Source section="A.5" pages="592–593" />
      <p>
        Every statement here about a vendor API or a codebase was read in the facts pass on 2026-11-20 and cites its item
        in <code>{FACTS}</code>. Projects like llama.cpp change weekly, so the item letters tell you where to look again.
      </p>

      <h2>What you run today</h2>
      <p>
        You have probably quantised a model already. Here is where three common routes end on the GPU, for one token at a
        time. In PyTorch that route is torchao now: <code>torch.ao.quantization</code> is superseded.
      </p>
      <Today checked={`2026-11-20 (${FACTS}, D6)`}>
        <p>PyTorch's 2.15 page still plans the deletion "in 2.10". Read that date as stale.</p>
      </Today>
      <table className="data wide" data-table="today" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr>
            <th>What you call</th>
            <th>What runs at decode</th>
            <th>Facts</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>torchao <code>Int4WeightOnlyConfig</code> (groups of 32 to 256, default 128); <code>Int8WeightOnlyConfig</code></td>
            <td>
              PyTorch's weight-only core ops are <code>_weight_int4pack_mm</code> (CUDA with compute capability 8.0 or a
              ROCm build; MPS) and <code>_weight_int8pack_mm</code> (CPU, CUDA, MPS). Which kernel a torchao config
              reaches depends on its packing format: not checked
            </td>
            <td>D6</td>
          </tr>
          <tr>
            <td><code>mlx.nn.QuantizedLinear</code> (defaults: group 64, 4 bits, affine)</td>
            <td>a quantised matrix-vector kernel on Metal, such as <code>affine_qmv_fast</code></td>
            <td>D5</td>
          </tr>
          <tr>
            <td>llama.cpp with a GGUF Q4_0 or Q8_0 tensor</td>
            <td>
              CUDA and HIP: <code>mmvq</code> for up to 8 tokens, after quantising the activations to Q8_1. Metal:{' '}
              <code>mul_mv</code>, in float
            </td>
            <td>D3, D4; ch15-facts D1</td>
          </tr>
        </tbody>
      </table>
      <p>
        These are four different kernels for one idea. Sort the calls below before reading on; the feedback for each one
        is a preview of a page of this module.
      </p>
      <KernelClassify />

      <h2>The size of the problem</h2>
      <p>
        The book puts a number on it: a model of 7 billion parameters in a 16-bit format takes about 14 GB. Decode at
        batch 1 reads all of it for every token. The book's own layer, 1024 to 4096 in 16-bit, does{' '}
        {bookLayer.intensity.toFixed(3)} FLOP per byte at batch 1.
      </p>
      <Source section="20.6" pages="505–506" />
      <Predict
        id="x2/start/p-q4-gb"
        unit="GB"
        prompt={
          <>
            The same 7 billion weights stored as llama.cpp's Q4_0: 4-bit codes in blocks of 32, each block with one 16-bit
            scale. How many GB?
          </>
        }
      >
        {(c) => (
          <div data-reveal="sizes">
            <p>
              <b>{gb(q4)} GB</b>
              {Math.abs(c - q4 / 1e9) < 0.05 ? ', which is your number' : Math.abs(c - 3.5) < 0.05 ? ': 3.5 GB is the codes alone' : ''}. Each block
              is 16 bytes of codes and 2 bytes of scale, 18 bytes for 32 weights, 4.5 bits per weight. One byte in nine of
              a Q4_0 file is scale, and you pay for it on every token.
            </p>
            <SizeTable />
            <p style={{ marginTop: '0.6rem' }}>
              Q8_0 is 8.5 bits, not 8, for the same reason. Section <a href={href('x2', 'X2.1')}>X2.1</a> turns these bytes
              into a predicted speed-up, and shows what a separate dequantise step does to it.
            </p>
          </div>
        )}
      </Predict>

      <h2>The plan</h2>
      <p>
        Seven pages. <a href={href('x2', 'X2.1')}>X2.1</a> counts bytes and predicts speed.{' '}
        <a href={href('x2', 'X2.2')}>X2.2</a> builds scales, zero points and blocks, and <a href={href('x2', 'X2.3')}>X2.3</a>{' '}
        packs two codes per byte. <a href={href('x2', 'X2.4')}>X2.4</a> maps a warp onto quantised rows.{' '}
        <a href={href('x2', 'X2.5')}>X2.5</a> compares dequantising to float with multiplying in integers, and{' '}
        <a href={href('x2', 'X2.6')}>X2.6</a> says where the sum must stay wide. <a href={href('x2', 'X2.7')}>X2.7</a> puts
        it together into a decode step.
      </p>
      <p>
        Four labs, on the <a href={href('x2', 'labs')}>labs page</a>, each with a prediction before you measure:
      </p>
      <ul>
        <li><b>Lab X2.1</b>, an int8-weight GEMV for batch-1 decode.</li>
        <li><b>Lab X2.2</b>, a 4-bit GEMV on llama.cpp's Q4_0 blocks, read as the file stores them.</li>
        <li><b>Lab X2.3</b>, a bug hunt on the integer path.</li>
        <li><b>Lab X2.4</b>, milestone M9: the quantised decode step, with its bandwidth utilisation reported.</li>
      </ul>

      <h2>Before you start</h2>
      <p>
        Six questions on what this module assumes from chapters 5, 6, 10, 20 and 22 and appendix A. Each links to the
        section it comes from, and each comes back in <a href={href('review')}>review</a>.
      </p>
      <Ask q={q['x2/start/q-gemv-intensity']} label="From chapters 5 and 20" />
      <Ask q={q['x2/start/q-gather']} label="From chapter 22" />
      <Ask q={q['x2/start/q-coalesce']} label="From chapter 6" />
      <Ask q={q['x2/start/q-fp16-int']} label="From appendix A" />
      <Ask q={q['x2/start/q-acc-order']} label="From appendix A" />
      <Ask q={q['x2/start/q-warp-reduce']} label="From chapter 10" />
    </>
  );
}
