import { useMemo } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Beyond, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { TEASER_N, pct, show, teaser, ulpText } from '../../sim/fpsum';

const FACTS = 'notes/today/appA-facts.md';

// ---------------------------------------------------------------- the teaser's reveal
function TeaserReveal({ v }: { v: number }) {
  const t = useMemo(() => teaser(), []);
  const half = t.rows.find((r) => r.id === 'fp16-loop')!.e;
  const off = Math.abs(v - half.rel * 100);
  return (
    <>
      <table className="data" style={{ background: 'var(--page)', margin: '0.4rem 0 0.8rem' }} aria-label="Four ways to sum one softmax denominator">
        <thead>
          <tr>
            <th>Accumulator and order</th>
            <th>Result</th>
            <th>Relative error</th>
            <th>ULPs of the accumulator</th>
          </tr>
        </thead>
        <tbody>
          {t.rows.map((r) => (
            <tr key={r.id} data-row={r.id}>
              <td>{r.label}</td>
              <td data-k="result">{show(r.e.result, 9)}</td>
              <td data-k="rel">{pct(r.e.rel)}</td>
              <td data-k="ulps">{ulpText(r.e.ulps)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>
        The reference, a compensated sum in double precision, is <b data-k="reference">{show(t.reference, 10)}</b>. The half
        loop is off by <b>{pct(half.rel)}</b>
        {off < 10 ? ', close to what you predicted.' : v < half.rel * 100 ? ', far more than you predicted.' : ', less than you predicted.'}{' '}
        The same terms in the same half format, added as a tree, land within one unit in the last place.
      </p>
      <p>
        The data did not change between the rows; the accumulator and the order did. The half loop stalls: once its
        running sum is a few hundred, most terms are smaller than half the gap between neighbouring half values, and
        each addition rounds them away. Section <a href={href('appA', 'A.4')}>A.4</a> shows one such addition, and
        section <a href={href('appA', 'A.6')}>A.6</a> shows what to do about the loop.
      </p>
    </>
  );
}

// ---------------------------------------------------------------- prerequisite questions
export const questionsSA0: Record<string, Question> = {
  'appA/start/q-tolerance': {
    kind: 'choice',
    id: 'appA/start/q-tolerance',
    prompt: (
      <>
        A unit test compares a GPU tree sum of a million floats with a CPU loop over the same floats, using{' '}
        <code>==</code>. It fails: the two differ by 3 ULPs. What is the right fix?
      </>
    ),
    options: [
      {
        label: 'Compare both with an accurate reference, within a tolerance scaled to the sum of |x|.',
        correct: true,
        why: 'Neither order gives the exact sum, so neither can be the oracle for the other.',
      },
      { label: 'Make the kernel add the elements in index order, so both sides round alike.', why: 'That gives up the parallelism, and the loop is not exact either.' },
      { label: 'Accumulate in double on the GPU, so the low bits no longer differ.', why: 'It shrinks the gap without closing it, and Metal has no double at all.' },
      { label: 'Compare the bit patterns of the two results instead of their values.', why: 'The bit patterns differ exactly where the values do.' },
    ],
    explain: (
      <p>
        Floating-point addition is not associative, so a tree and a loop round differently and both are legitimate
        results. A test should accept any answer within a tolerance of an accurate reference. The labs already do this.
      </p>
    ),
    source: `ch. 10 §10.2, p. 224 (and appendix A §A.6, p. 593); Metal has no double: ${FACTS} C1`,
  },
  'appA/start/q-convergent': {
    kind: 'numeric',
    id: 'appA/start/q-convergent',
    prompt: (
      <>
        Chapter 10's convergent kernel (Fig. 10.8) reduces 16 elements with 8 threads. In the first iteration, which
        element does thread 2 add to its own?
      </>
    ),
    answer: 10,
    unit: '(element index)',
    explain: (
      <p>
        Thread i owns element i, and the stride starts at the block size, 8: thread i adds element i + 8. On sorted data
        that pairs a small value with a large one. It comes back in section A.6 as the book's exercise 4.
      </p>
    ),
    source: 'ch. 10 §10.4, pp. 232–233',
  },
  'appA/start/q-bytes': {
    kind: 'numeric',
    id: 'appA/start/q-bytes',
    prompt: (
      <>
        A matrix-vector product at batch size 1 stores its weights in 1 byte each. Counting a multiply and an add per
        weight, how many operations does it perform per byte of weight traffic?
      </>
    ),
    answer: 2,
    unit: 'FLOP/B',
    explain: (
      <p>
        Two operations per weight, one byte per weight. Chapter 20's linear layer at batch 1 with 16-bit weights gets 1
        FLOP/B; halving the bytes doubles it. Section A.5 adds what the format costs on top: the scales stored with each
        block of weights.
      </p>
    ),
    source: 'ch. 5 §5.1, p. 94 (FLOP/B); ch. 20 §20.6, p. 505 (1 FLOP/B at batch 1, 16-bit)',
  },
  'appA/start/q-atomic': {
    kind: 'choice',
    id: 'appA/start/q-atomic',
    prompt: (
      <>
        A multi-block float sum commits each block's partial sum with an atomic add. You run it 20 times on the same input.
        Can the results differ?
      </>
    ),
    options: [
      {
        label: 'Yes: blocks commit in a different order each run, and float addition is not associative.',
        correct: true,
      },
      { label: 'No: atomic operations make each update indivisible, so the total is always the same.', why: 'Indivisible updates still arrive in any order. Atomics do not fix the order.' },
      { label: 'Only if two blocks happen to commit at exactly the same moment.', why: 'Simultaneous commits are serialised either way; it is the order that varies.' },
      { label: 'Yes, and an integer sum would differ from run to run in the same way.', why: 'Integer addition without overflow is associative: any order gives the same total.' },
    ],
    explain: (
      <p>
        An atomic add gives mutual exclusion, not ordering, which is why the operator must be associative and
        commutative. For floats that holds only up to rounding, so the low bits of the total can change from run to run.
      </p>
    ),
    source: 'ch. 10 §10.9, p. 245',
  },
};

// ---------------------------------------------------------------- the page
export function SA0() {
  return (
    <>
      <p>
        One decoder step of an LLM is, among other things, thousands of long sums. Every dot product inside a GEMM is
        one; so is the softmax normaliser over a row of logits, and the mean of squares a norm takes over the hidden
        dimension. Each runs in finite precision, into an accumulator whose format somebody chose. This appendix gives
        you what you need to make that choice yourself and to predict what it costs.
      </p>
      <Beyond title="Not in the book: who chooses the accumulator">
        <p>
          PyTorch accumulates half, bfloat16 and FP8 in float on CUDA, MPS and CPU, and float in double on the CPU. MLX's
          default softmax accumulates in the input type unless <code>precise=True</code>; by our reading of its source,{' '}
          <code>mx.sum</code> on half accumulates in half. llama.cpp states a required accumulator per operation with{' '}
          <code>ggml_prec_set_acc</code> ({FACTS} D1 to D3).
        </p>
      </Beyond>

      <h2>One sum, four ways</h2>
      <p>
        Take one softmax denominator: a row of {TEASER_N.toLocaleString('en-US')} logits, drawn from a standard normal
        distribution, and the sum of exp(z − max z) over the row. Every term lies in (0, 1]. Sum it four ways: a loop and a
        pairwise tree, each with a float (FP32) and with a half (FP16) accumulator, and compare each with an accurate
        reference.
      </p>
      <Beyond>
        <p>
          The row and every number in the reveal come from our model of the four sums (round to nearest even after every
          addition; nothing about a particular GPU). Lab A.1 measures summation error on yours.
        </p>
      </Beyond>
      <Predict
        id="appA/start/p-teaser"
        prompt={
          <>
            How far off do you expect the half-precision <b>loop</b> to be, as a percentage of the true sum? Commit to a
            number before the four results appear.
          </>
        }
        unit="%"
      >
        {(v) => <TeaserReveal v={v} />}
      </Predict>

      <h2>What this appendix is about</h2>
      <p>
        The book's appendix starts from why floating point is worth the trouble: for the same number of bits it covers a
        far wider range of values than an integer format, and it represents small values finely. It then separates
        three ideas that are easy to blur. <b>Precision</b> is a property of a format: how close its nearest value can be
        to the number you want to store. <b>Accuracy</b> is a property of an operation: how much error one addition or
        division adds. <b>Stability</b> is a property of an algorithm: whether it finds a solution for every input that
        has one (section <a href={href('appA', 'A.7')}>A.7</a>).
      </p>
      <Source pages="583, 594" />
      <p>
        The book also notes why this matters more now than it did: demand from LLM training and inference has pushed GPUs
        to raise their floating-point throughput dramatically. Section A.5 adds that recent GPUs compute in 16-, 8- and
        4-bit floating point.
      </p>
      <Source pages="583, 593" />
      <Today checked={`2026-11-20 (${FACTS}, A1, A4)`}>
        <p>
          The footnote's OCP MX formats (cited as v1.0) are now the "OCP MicroXcaling (MX) Specification" v1.1 (October
          9, 2026), with E8M0 renamed UE8M0. All three vendors use them: NVIDIA's PTX <code>.kind::mxf8f6f4</code> and{' '}
          <code>.kind::mxf4</code> and cuBLAS VEC32_UE8M0, AMD gfx950's <code>__amd_scale_t</code>, Metal 4.1's{' '}
          <code>tensor_blockwise</code>.
        </p>
        <p>IEEE 754-2019 is still the active standard; P754 (a revision) and P3109 (ML formats) are open projects.</p>
      </Today>

      <h2>The map</h2>
      <ul>
        <li>
          <a href={href('appA', 'A.1')}>A.1</a> to <a href={href('appA', 'A.3')}>A.3</a>: how a format stores a value,
          which values it can hold, and its special values (zero, denormals, infinity, NaN).
        </li>
        <li>
          <a href={href('appA', 'A.4')}>A.4</a>: the error of one operation, in units in the last place.
        </li>
        <li>
          <a href={href('appA', 'A.5')}>A.5</a>: narrow formats and block scaling, the starting point for the quantised
          kernels of module X2.
        </li>
        <li>
          <a href={href('appA', 'A.6')}>A.6</a>: long sums, the core of this appendix for inference kernels. It ties back to
          chapter 10's reductions and milestone M4's softmax.
        </li>
        <li>
          <a href={href('appA', 'A.7')}>A.7</a>: numerical stability, condensed.
        </li>
        <li>
          <a href={href('appA', 'labs')}>Lab A.1</a> measures summation error, by order, accumulator width and
          compensation, on your GPU.
        </li>
      </ul>

      <h2>Before you start</h2>
      <p>
        Four questions on what this appendix assumes from chapters 5, 10 and 20. Each one comes back in{' '}
        <a href={href('review')}>review</a>.
      </p>
      <Ask q={questionsSA0['appA/start/q-tolerance']} label="From chapter 10" />
      <Ask q={questionsSA0['appA/start/q-convergent']} label="From chapter 10" />
      <Ask q={questionsSA0['appA/start/q-bytes']} label="From chapters 5 and 20" />
      <Ask q={questionsSA0['appA/start/q-atomic']} label="From chapter 10" />
    </>
  );
}
