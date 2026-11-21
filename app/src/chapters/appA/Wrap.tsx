import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';

// The numbers every question below states are pinned by Wrap.test.ts, computed with the
// same models (sim/minifloat.ts, sim/fpsum.ts) the sections use.
const FACTS = 'notes/today/appA-facts.md';

export const questionsWrap: Record<string, Question> = {
  'appA/wrap/q-rms-half': {
    kind: 'choice',
    id: 'appA/wrap/q-rms-half',
    prompt: (
      <>
        A norm layer sums the squares of 4,096 activations to get their mean. Every activation is exactly 4, so each
        square is 16 and the true sum is 65,536. The sum is kept in FP16. What do a plain loop and a halving tree (each
        level adds pairs) return?
      </>
    ),
    options: [
      {
        label: 'The loop returns 32,768; the tree returns infinity.',
        correct: true,
      },
      {
        label: 'Both return 65,536: every partial sum is a whole number.',
        why: 'Whole numbers above 2,048 are not all FP16 values, and 65,536 is above the largest one, 65,504.',
      },
      {
        label: 'Both return infinity: the true sum is too large for FP16.',
        why: 'The loop never gets there. At 32,768 the gap between FP16 values is 32, and adding 16 is exactly half a gap.',
      },
      {
        label: 'The loop returns 65,504 and the tree returns infinity.',
        why: 'The loop stops growing well before that. Where, and why exactly there?',
      },
    ],
    explain: (
      <p>
        Up to 32,768 every multiple of 16 is an FP16 value, so the loop is exact until then. Above 32,768 the spacing is
        32, and 32,768 + 16 lies exactly between two neighbours; round to nearest even picks 32,768, and every later
        addition does the same. The tree's partial sums double at each level, all exact, until the last addition,
        32,768 + 32,768, overflows. The loop's mean is 8 where it should be 16, so the root mean square comes out about
        2.83 instead of 4 and every output is about 1.41 times too large. The tree's infinity makes every output 0.
        Accumulating in float fixes both, which is what PyTorch does for half inputs on every device.
      </p>
    ),
    source: `ours, from §A.2 (spacing, pp. 586–590), §A.3 (overflow to infinity, p. 590) and §A.6 (p. 593); ${FACTS} D1`,
  },
  'appA/wrap/q-bf16-argmax': {
    kind: 'numeric',
    id: 'appA/wrap/q-bf16-argmax',
    prompt: (
      <>
        Greedy decoding takes the argmax of a row of logits. A kernel casts the logits to BF16 first. Two candidates have
        FP32 logits of 10.01 and 10.03. What is the gap between neighbouring BF16 values near 10?
      </>
    ),
    answer: 0.0625,
    tol: 0.001,
    explain: (
      <p>
        BF16 keeps 7 stored mantissa bits, so between 8 and 16 its values are 2<sup>3</sup> × 2<sup>−7</sup> = 0.0625
        apart. Both logits round to 10.0, and the argmax now has to break a tie that FP32 did not have; which token
        wins depends on its tie rule. The greedy token can change. Take the argmax on the FP32 logits, or on whatever the last GEMM
        accumulated in, before any cast.
      </p>
    ),
    source: 'ours, from §A.2 (precision, pp. 586–590); BF16 field widths: notes/today/appA-facts.md A2',
  },
  'appA/wrap/q-logprob-ftz': {
    kind: 'choice',
    id: 'appA/wrap/q-logprob-ftz',
    prompt: (
      <>
        A sampler needs log-probabilities. Your device flushes FP32 denormals to zero (lab A.1's probe tells you whether
        yours does). A token's logit is 90 below the row's maximum. Which way of computing its log-probability gives a
        finite answer there?
      </>
    ),
    options: [
      {
        label: 'Take the log of the softmax probability as written, log(exp(z − max) / Σ), in float.',
        why: 'exp(−90) is about 8.2 × 10⁻⁴⁰, below the smallest normal float. Flushed, it is 0, and log 0 is minus infinity.',
      },
      {
        label: 'Compute (z − max) − log Σ exp(zⱼ − max), without forming the probability.',
        correct: true,
      },
      {
        label: 'Either: exp(−90) is an ordinary FP32 value.',
        why: 'Only as a denormal, which this device does not keep.',
      },
      {
        label: 'Neither: −90 is outside FP32\'s range.',
        why: '−90 itself is an ordinary float. Only its exponential is tiny.',
      },
    ],
    explain: (
      <p>
        exp(−90) is about 8.2 × 10<sup>−40</sup>. FP32's smallest normal value is about 1.18 × 10<sup>−38</sup>, so the
        probability exists only as a denormal. On a device that keeps denormals both forms work; on one that flushes
        them, the first gives minus infinity, and a later sum or a NaN test may then behave in ways that are hard to
        trace. The second form never computes the tiny number: its terms are about −90 and the log of a sum that is at
        least 1.
      </p>
    ),
    source: `ours, from §A.2 (denormals, pp. 588–589); flushing: ${FACTS} A6, C2`,
  },
  'appA/wrap/q-e4m3-small': {
    kind: 'numeric',
    id: 'appA/wrap/q-e4m3-small',
    prompt: (
      <>
        A KV cache stores activations in E4M3 with no scale factor. One activation is 0.0006. What value is stored?
      </>
    ),
    answer: 0,
    explain: (
      <p>
        E4M3's smallest positive value is the denormal 2<sup>−9</sup>, about 0.00195. 0.0006 is less than half of it,
        so it rounds to 0. Multiplying by a power of two first, say 2<sup>10</sup>, moves it to about 0.61, where E4M3
        keeps it to within a few percent; the scale is stored beside the block and divided out when the cache is read.
        That is the reason narrow formats come with scales, for activations as much as for weights.
      </p>
    ),
    source: `ours, from §A.2 (gap near zero, pp. 588–589) and §A.5 (pp. 592–593); E4M3: ${FACTS} A3`,
  },
  'appA/wrap/q-batch-order': {
    kind: 'choice',
    id: 'appA/wrap/q-batch-order',
    prompt: (
      <>
        Your server returns slightly different logits for the same prompt when it runs alone and when it runs in a batch
        of 8. Each configuration gives the same logits every time it runs. The GEMM library chooses a different split of
        the k dimension for each batch size. What is the most likely explanation?
      </>
    ),
    options: [
      {
        label: 'The two splits add each dot product\'s partial products in different orders, and float addition is not associative.',
        correct: true,
      },
      {
        label: 'A race condition between the blocks that add the partial sums of the k dimension.',
        why: 'A race would make one configuration differ from run to run. Here each one is repeatable.',
      },
      {
        label: 'The batched GEMM flushes denormals and the single one does not.',
        why: 'Flushing is a property of the device and the compiler settings, not of the batch size.',
      },
      {
        label: 'One of the two results is wrong: a correct GEMM gives the same bits at every batch size.',
        why: 'Both orders are legitimate roundings of the same sum. Neither is the exact answer.',
      },
    ],
    explain: (
      <p>
        Splitting k into more pieces turns one long sum into several shorter ones added at the end: a different tree.
        Section A.6's point is that every such tree is a valid result and they differ in the low bits. If you need the
        same logits whatever the batch, the reduction order of each output must not depend on the batch size, which
        usually costs some speed.
      </p>
    ),
    source: 'book §A.6, p. 593 (order changes the sum); the GEMM setting is ours',
  },
};

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = realBackends(reports);
  const lab = backends.filter((b) => labStatus(reports[b], 'appA/lab1_sumerror', 'work') === 'pass');
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('appA/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul>
        <li>
          A format is a sign, an exponent and a mantissa. Exponent bits buy range, mantissa bits buy precision, and the
          gap between neighbouring values doubles at every power of two.
        </li>
        <li>
          Near zero a format either has a gap or fills it with denormals. Every backend lets denormals be lost somewhere:
          a compiler flag, a fast-math mode, or the device itself. Lab A.1's probe tells you what yours does.
        </li>
        <li>
          Infinity and NaN are values, not errors. Nothing on a GPU stops when one appears, so a softmax subtracts the
          maximum first and a test checks its outputs.
        </li>
        <li>
          Precision belongs to the format, accuracy to the operation. Basic arithmetic is correctly rounded where the
          vendor says so; fast-math functions and some backends' rounding modes are not.
        </li>
        <li>
          Narrow formats trade precision for bytes, and a memory-bound decode step turns fewer bytes per weight into
          speed. A shared scale per block recovers the range.
        </li>
        <li>
          A long sum depends on its order and on its accumulator. A tree, a sorted order, Kahan's compensation and a
          wider accumulator each help; a narrow accumulator stalls. Compare GPU results with an accurate reference and a
          tolerance, never with another order.
        </li>
        <li>
          Stability is a property of the algorithm. Pivoting keeps elimination stable, and in parallel it costs a max
          reduction per column.
        </li>
      </ul>
      <Source section="A.8" pages="599" />
      <p>
        The book's own summary adds one line worth keeping: early CUDA devices lacked denormals and later ones have them,
        and the fast operations of the "special function units" are less accurate than the ordinary ones.
      </p>
      <Today checked={`2026-11-20 (${FACTS} A6, A14)`}>
        <p>
          NVIDIA keeps FP32 denormals by default and flushes them under <code>--ftz=true</code> or{' '}
          <code>--use_fast_math</code>. Its current documentation does not use the term "special function units"; it
          says intrinsics are faster because they "map to fewer native instructions".
        </p>
      </Today>

      <h2>A mixed check</h2>
      <p>
        These draw on the whole appendix and on situations the sections did not show.
        {asked ? ` So far in this appendix you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
      </p>
      <Ask q={questionsWrap['appA/wrap/q-rms-half']} />
      <Ask q={questionsWrap['appA/wrap/q-bf16-argmax']} />
      <Ask q={questionsWrap['appA/wrap/q-logprob-ftz']} />
      <Ask q={questionsWrap['appA/wrap/q-e4m3-small']} />
      <Ask q={questionsWrap['appA/wrap/q-batch-order']} />

      <h2>The book's exercises</h2>
      <p>
        The appendix ends with five exercises (pp. 599–600). Each is worked somewhere in this course; the book's own
        wording is worth doing as well.
      </p>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-table="exercises">
          <thead>
            <tr>
              <th>Exercise</th>
              <th>About</th>
              <th>Where it is worked here</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1</td>
              <td>A 6-bit format with a third mantissa bit</td>
              <td>
                the number-line explorer and its question in <a href={href('appA', 'A.2')}>A.2</a>
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td>A 6-bit format with a third exponent bit</td>
              <td>
                the same explorer and its second question in <a href={href('appA', 'A.2')}>A.2</a>
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td>The largest error of an adder that only truncates</td>
              <td>
                the alignment stepper set to "toward zero", and its question, in <a href={href('appA', 'A.4')}>A.4</a>;
                on CUDA, the truncating adder in <a href={href('appA', 'labs')}>lab A.1</a>'s "go further"
              </td>
            </tr>
            <tr>
              <td>4</td>
              <td>A sorted array reduced by the convergent tree</td>
              <td>
                the "thread to data, with rounding" stepper and its question in <a href={href('appA', 'A.6')}>A.6</a>
              </td>
            </tr>
            <tr>
              <td>5</td>
              <td>The ULP error of a sin() unit that stops early</td>
              <td>
                the question in <a href={href('appA', 'A.4')}>A.4</a>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        The book points to Goldberg's survey (its reference [1]) for more on floating point, and to Kahan's 1965 note
        ([5]) for compensated summation. Its reference [3] is now the OCP MicroXcaling (MX) specification, version 1.1,
        and its reference [4], NVIDIA's floating-point page, now serves the 2011 white paper: the maintained text is
        section 5.6 of the CUDA Programming Guide.
      </p>
      <Today checked={`2026-11-20 (${FACTS} A4, A12)`}>
        <p>
          MX v1.1 (October 9, 2026) renamed the specification and its scale type E8M0 to UE8M0. The book's NVIDIA URL
          resolves to the 2011 white paper, which the guide's section 5.6 lists among its references.
        </p>
      </Today>

      <h2>Where the PDF looks wrong</h2>
      <p>
        The copy this course was built from has a number of apparent slips in this appendix. They are listed so that you
        are not thrown by them when you read the book alongside. Most are discussed where they come up.
      </p>
      <ul>
        <li>
          p. 589 says denormals are switched "on or off" with "ftz = true (on) or ftz = false (off)". The flag means flush
          to zero, so <code>--ftz=true</code> turns denormals off. See <a href={href('appA', 'A.2')}>A.2</a>.
        </li>
        <li>
          p. 588 places the missing zero in "the no-zero column of Fig. A.5"; the columns are in Fig. A.4. p. 589 points to
          Fig. A.6 for denormals, where Fig. A.7 is meant, and its "previous interval" is the next larger one. Fig. A.7
          draws a 13th star near 4, which the format cannot hold, and Fig. A.5 is not drawn to scale between 0 and 1. See{' '}
          <a href={href('appA', 'A.2')}>A.2</a>.
        </li>
        <li>
          p. 590 says three more exponent bits give "2<sup>(2<sup>3</sup>)</sup> = 256 times" as many intervals; they give
          2<sup>3</sup> = 8 times as many. See <a href={href('appA', 'A.3')}>A.3</a>.
        </li>
        <li>
          p. 591 gives single precision "2<sup>22</sup> − 1 possible NaN bit patterns". That is the number of signalling
          NaNs of one sign; there are 2<sup>24</sup> − 2 in all. See <a href={href('appA', 'A.3')}>A.3</a>.
        </li>
        <li>
          p. 585 argues the reserved pattern 111 cannot be "either even number or odd number"; positive or negative seems
          to be meant.
        </li>
        <li>
          p. 593 calls H100's 989 TFLOPS "single-precision throughput using tensor cores". NVIDIA's 989 is the TF32 row,
          with sparsity; the 3,958 FP8 figure beside it is also a with-sparsity number. See <a href={href('appA', 'A.5')}>A.5</a>.
        </li>
        <li>
          Fig. A.11's Step 1 caption says "divide row 1 by 3" where the row is divided by 2, and its Steps 5 and 6 print
          row 1 of Fig. A.10's example. Fig. A.10's Step 5 caption says "equation" where the others say "row". See{' '}
          <a href={href('appA', 'A.7')}>A.7</a>.
        </li>
        <li>
          pp. 598–599 define "partial pivoting" as choosing among a localized set of equations. In LAPACK, cuSOLVER,
          rocSOLVER and MPS it means swapping rows within the current column; swapping rows and columns is complete
          pivoting. See <a href={href('appA', 'A.7')}>A.7</a>.
        </li>
        <li>
          Exercise 4 refers to "the algorithm of Fig. A.4", which is a table. We take chapter 10's convergent tree to be
          meant. See <a href={href('appA', 'A.6')}>A.6</a>.
        </li>
        <li>
          Small things: Exercises 1 and 2 list the fields as sign, mantissa, exponent; Figs. A.2 and A.3 share a caption;
          and there are typos on pp. 583, 585, 586, 590, 591, 592 and 597.
        </li>
      </ul>
      <Aside>
        <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
      </Aside>

      <h2>The project so far</h2>
      <p>
        This appendix adds no milestone of its own. What it adds is the evidence for a decision two milestones depend
        on: which accumulator a kernel uses. Lab A.1 measured, on your device, what a half or bfloat16 accumulator does
        to a long sum and what Kahan's compensation costs.{' '}
        {backends.length === 0
          ? 'No lab results are recorded on this machine yet.'
          : lab.length
            ? `Your own version passes on ${lab.join(' and ')} on this machine.`
            : 'Your own version has not passed on this machine yet.'}{' '}
        See the <a href={href('appA', 'labs')}>labs</a>.
      </p>
      <p>
        Module <a href={href('x2')}>X2</a> picks this up. It writes int8- and 4-bit-weight GEMVs, where the weights are
        narrow and the accumulator must not be, and puts them into milestone M9, the quantised decode step, which
        measures what the fewer bytes buy. Milestone M4's softmax, lab
        10.3, keeps its sum in float for the reasons sections <a href={href('appA', 'A.3')}>A.3</a> and{' '}
        <a href={href('appA', 'A.6')}>A.6</a> gave.
      </p>
      <Beyond>
        <p>
          Integer quantisation, with its scales and zero points, and tensor-core or matrix-unit code for narrow types
          are not in the book. Module X2 covers them.
        </p>
      </Beyond>
    </>
  );
}
