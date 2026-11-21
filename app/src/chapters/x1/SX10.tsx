import type { CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Beyond, NoteGroup, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { DRAW, GREEDY, M5_ORDER, VOCAB, rowBytes, stage, type KernelStep } from '../../sim/chain';

// Module X1's start page: where the sampler sits, the chain as a map of the earlier labs
// that supply its kernels (drawn from chain.ts's STAGES, the same list the X1.1 explorer
// runs), what the reader's own stack does today, the size of the problem, what "same seed,
// same token" will mean, and a prerequisite check drawn from chapters 9 to 14.

const n = (v: number) => v.toLocaleString('en-US');
const FACTS = 'notes/today/x1-facts.md';

export const questionsSX10: Record<string, Question> = {
  'x1/start/q-max-subtract': {
    kind: 'choice',
    id: 'x1/start/q-max-subtract',
    prompt: (
      <>
        Lab 10.3's softmax subtracts the row maximum before it calls <code>exp</code> (
        <a href={href('ch10', 'labs')}>lab 10.3</a>). A row of logits reaches 95. What goes wrong in float32 if you
        skip the subtraction?
      </>
    ),
    options: [
      {
        label: 'exp(95) overflows to infinity, so the sum is infinite and every probability comes out as 0 or NaN.',
        correct: true,
      },
      {
        label: 'Nothing: float32 reaches about 3.4 × 10³⁸, which is more than exp(95).',
        why: 'exp(95) is about 1.8 × 10⁴¹. exp passes the largest finite float32 at an argument of about 88.7.',
      },
      {
        label: 'The probabilities lose a few digits of precision but stay usable.',
        why: 'Overflow is not a loss of digits. The result is infinity, and infinity divided by infinity is NaN.',
      },
      {
        label: 'Only the largest token is affected: it gets probability 1, the rest get 0, and greedy decoding still picks it.',
        why: 'Its numerator and the sum are both infinite, and their quotient is NaN, not 1.',
      },
    ],
    explain: (
      <p>
        Subtracting the row maximum m leaves every probability unchanged (numerator and denominator are both
        multiplied by e<sup>−m</sup>) and makes every exponent at most 0. The largest term is then exactly 1 and the sum is
        at least 1, so nothing overflows and the division is safe. Lab 10.3 tests rows offset by +90 for exactly this.
      </p>
    ),
    source: 'ch. 20 §20.2, Eq. 20.2, p. 484; lab 10.3 README',
  },
  'x1/start/q-exclusive': {
    kind: 'numeric',
    id: 'x1/start/q-exclusive',
    prompt: (
      <>
        Sorted probabilities 0.5, 0.25, 0.125, 0.125 and p = 0.75. Lab 11.6's rule keeps token i when the mass ranked
        before it is below p (<a href={href('ch11', '11.1')}>section 11.1</a>). How many tokens does it keep?
      </>
    ),
    answer: 2,
    unit: 'tokens',
    explain: (
      <p>
        The mass before each token is 0, 0.5, 0.75 and 0.875. Tokens 0 and 1 start below 0.75 and are kept; token 1's own
        share brings the total to exactly 0.75, so the set already reaches p and token 2, which starts at 0.75, is not
        needed. This is the exclusive scan of section 11.1, and it is the rule llama.cpp's samplers compute.
      </p>
    ),
    source: 'ch. 11 §11.1, p. 253; notes/today/ch11-facts.md D2',
  },
  'x1/start/q-stable': {
    kind: 'choice',
    id: 'x1/start/q-stable',
    prompt: (
      <>
        Lab 12.4 builds each row's candidate list with a stable filter, a scan of keep flags (
        <a href={href('ch12', '12.5')}>section 12.5</a>), rather than with an atomic counter. Why does a sampler care?
      </>
    ),
    options: [
      {
        label: 'An atomic counter hands out slots in arrival order, so the list, and any tie broken by position, can change from run to run.',
        correct: true,
      },
      { label: 'Atomics are too slow for a vocabulary-sized row, where thousands of threads hit one counter.', why: 'Speed is not what rules them out here. Order is.' },
      { label: 'Metal has no atomics on device memory, and the filter must run on all three backends, so it has to use a scan.', why: 'Metal has 32-bit integer atomics (notes/today/ch09-facts.md C1).' },
      { label: 'A scan needs less memory than a counter.', why: 'A scan needs more: an offset per element against one counter per row.' },
    ],
    explain: (
      <p>
        The order of an atomic filter's output is whatever order the threads happened to arrive in. A sampler that breaks
        ties by token id, and that you want to give the same token for the same seed, needs the same list every run.
        A stable filter writes each kept token at the number of kept tokens before it, which depends only on the data.
      </p>
    ),
    source: 'ch. 12 §12.5, pp. 297–298; lab 12.4 README',
  },
  'x1/start/q-vocab-kib': {
    kind: 'numeric',
    id: 'x1/start/q-vocab-kib',
    prompt: (
      <>
        One <code>uint32</code> count per token for Qwen3-8B's vocabulary of {n(VOCAB.qwen3)} ids: how many KiB is one
        private copy of the counts (<a href={href('ch09', '9.4')}>section 9.4</a>)?
      </>
    ),
    answer: (VOCAB.qwen3 * 4) / 1024,
    tol: 0.001,
    unit: 'KiB',
    explain: (
      <p>
        {n(VOCAB.qwen3)} × 4 bytes = {n(VOCAB.qwen3 * 4)} bytes = 593.5 KiB. That is more than any documented per-block
        on-chip limit, which is why lab 9.3 counts straight into global memory with one row of counts per request and
        no shared-memory private copy.
      </p>
    ),
    source: 'ch. 9 §9.4, pp. 211–213, and Fig. 9.10, p. 214; notes/today/ch09-facts.md D7',
  },
  'x1/start/q-msd-bin': {
    kind: 'numeric',
    id: 'x1/start/q-msd-bin',
    prompt: (
      <>
        An MSD radix select for the 20 largest keys of a row reads its top-8-bit histogram from the highest bin down: 2,
        5, 30, 100, … keys (<a href={href('ch14', '14.9')}>section 14.9</a>). How many keys must it still pick from inside
        the bin where the 20th largest key lies?
      </>
    ),
    answer: 13,
    unit: 'keys',
    explain: (
      <p>
        The running totals are 2, 7, 37. The 20th key lies in the third bin, which holds 30 keys. The 7 keys above that
        bin are all in, so 13 more are needed from inside it, and the next pass looks only at those 30 keys, digit by
        digit. Lab X1.1 builds this selection; it never sorts the row.
      </p>
    ),
    source: 'ch. 14 §14.9, p. 346; notes/today/ch14-facts.md D2',
  },
  'x1/start/q-one-block': {
    kind: 'choice',
    id: 'x1/start/q-one-block',
    prompt: (
      <>
        Every kernel of milestone M5 runs one block per row. At batch 1, on a device with S multiprocessors, what share of
        the multiprocessors can one such kernel keep busy at most (<a href={href('ch10', '10.10')}>section 10.10</a>)?
      </>
    ),
    options: [
      { label: '1/S: the grid is one block, and a block runs on one multiprocessor', correct: true },
      { label: 'All of them, because a vocabulary-sized row is long enough to fill the device', why: 'The length of the row decides how long the one block works, not how many multiprocessors it occupies.' },
      { label: 'Half of them, with the rest held back for the next kernel in the stream', why: 'Nothing in the launch puts work on half the device.' },
      { label: 'It depends on the vocabulary size: a longer row spreads over more of them', why: 'The vocabulary sets how much work the block does. The grid has one block either way.' },
    ],
    explain: (
      <p>
        A block is assigned to one multiprocessor and stays there. One row gives one block, so the rest of the device has
        nothing to do. The ways up are more rows (a batch) or splitting a row across blocks and combining their results,
        as section 10.9 does. In M5 the batch is the grid.
      </p>
    ),
    source: 'ch. 4 §4.2, p. 69; ch. 10 §10.10, pp. 246–248',
  },
};

// ---------------------------------------------------------------- the map
const BOX: CSSProperties = {
  border: '1px solid var(--rule)',
  borderRadius: 4,
  background: 'var(--page)',
  padding: '0.6rem 0.7rem',
  fontFamily: 'var(--sans)',
  fontSize: '0.86rem',
  lineHeight: 1.4,
  margin: 0,
  minWidth: 0,
};
const LIST: CSSProperties = { listStyle: 'none', margin: '0.3rem 0 0', padding: 0 };

function KernelLine({ k }: { k: KernelStep }) {
  return (
    <li style={{ margin: '0 0 0.3rem' }} data-lab={k.lab.num}>
      {k.what}: <a href={href(k.lab.chapter, 'labs')}>lab {k.lab.num}</a>
    </li>
  );
}

/** The chain as M5 builds it, drawn from STAGES in M5's order, then the draw. */
function ChainMap() {
  return (
    <ol className="wide" data-part="chain-map" aria-label="The sampler chain and the labs that supply it"
      style={{ listStyle: 'none', padding: 0, margin: '0.4rem 0 1.4rem', display: 'grid', gap: '0.5rem', gridTemplateColumns: 'repeat(auto-fit, minmax(12.5rem, 1fr))', maxWidth: '62rem' }}>
      {M5_ORDER.map((id, i) => {
        const s = stage(id);
        return (
          <li key={id} style={BOX} data-stage={id}>
            <b style={{ fontSize: '0.95rem' }}>{i + 1}. {s.label}</b>{' '}
            <span className="muted">{s.kind === 'truncate' ? 'removes candidates' : 'changes logits'}</span>
            <ul style={LIST}>{s.kernels.map((k) => <KernelLine key={k.what} k={k} />)}</ul>
          </li>
        );
      })}
      <li style={{ ...BOX, borderColor: 'var(--ink-3)' }} data-stage="draw">
        <b style={{ fontSize: '0.95rem' }}>{M5_ORDER.length + 1}. draw</b> <span className="muted">one token id per row</span>
        <ul style={LIST}>
          <KernelLine k={DRAW} />
          <li className="muted" style={{ margin: 0 }} data-lab={GREEDY.lab.num}>
            at T ≤ 0 instead: {GREEDY.what}, <a href={href(GREEDY.lab.chapter, 'labs')}>lab {GREEDY.lab.num}</a>
          </li>
        </ul>
      </li>
    </ol>
  );
}

export function SX10() {
  const q = questionsSX10;
  const one = rowBytes(VOCAB.qwen3);
  return (
    <>
      <p>
        The last matrix multiplication of a decode step leaves you with logits: a <code>[batch × vocabulary]</code> tensor
        of float32 scores, one row per request. The next kernel has to turn each row into one token id. That kernel is the
        sampler, and this module builds it on the GPU, on CUDA, HIP and Metal, from kernels you have already written.
      </p>
      <NoteGroup>
        <Beyond title="Ours, not the book's">
          <p>
            The book's transformer ends at the un-embedding layer, which turns the last layer's output into a probability
            distribution over tokens. How one token is chosen is not in the book. Module X1 is ours.
          </p>
        </Beyond>
        <Source section="20.1" pages="481" />
      </NoteGroup>
      <p>
        A sampler is a short chain of row-wise operations: penalise tokens that were already used, keep only the most
        likely candidates in three different ways, sharpen or flatten what is left, and draw one. Each operation is small.
        The interesting parts are the order they run in, the tie rules, and making the same seed give the same token on
        three backends.
      </p>

      <h2>The chain, and who supplies each stage</h2>
      <p>
        Milestone M5 runs the stages below, in this order, one block per row. Every box names the earlier lab whose kernel
        it reuses. Three new labs fill the gaps: a top-k that selects instead of sorting (X1.1), a min-p filter and a fused
        top-p and min-p kernel (X1.2), and the draw with its random numbers (X1.3). Section{' '}
        <a href={href('x1', 'X1.1')}>X1.1</a> explains why this order and not another.
      </p>
      <Aside>
        <p>
          The map is drawn from the same stage list the explorer of X1.1 runs, so the two cannot disagree. llama.cpp's
          chain has four more stages that M5 does not build; X1.1 shows them.
        </p>
      </Aside>
      <ChainMap />

      <h2>What you run today</h2>
      <p>
        If you call <code>generate</code> in Hugging Face transformers, the logits processors run as tensor operations
        and the token is drawn by <code>torch.multinomial</code>, on whatever device the logits are on. llama.cpp samples
        on the host by default, with <code>std::mt19937</code> and an inverse-CDF walk; it has an experimental{' '}
        <code>--backend-sampling</code> path that is on by default only for the drafts of speculative decoding. mlx-lm
        runs every filter as MLX operations, so on the GPU under Metal. vLLM samples on the GPU, with a Triton kernel that
        finds a pivot instead of sorting, and FlashInfer as an option on CUDA.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${FACTS}, D1, D3, D4, D5, D8; notes/today/ch14-facts.md D7)`}>
        <p>
          llama.cpp: <code>bool backend_sampling = false;</code> for generation, <code>true</code> for the speculative
          draft. transformers: <code>torch.multinomial(probs, num_samples=1)</code>. mlx-lm: filters are MLX ops. vLLM: a
          pivot-based Triton kernel by default; <code>flashinfer.sampling</code> optional.
        </p>
      </Today>

      <h2>The size of the problem</h2>
      <Predict
        id="x1/start/p-bytes"
        unit="bytes"
        prompt={
          <>
            One sampling step for one row of Qwen3-8B: {n(VOCAB.qwen3)} float32 logits, plus the {n(VOCAB.qwen3)}{' '}
            <code>uint32</code> token counts the penalties need. What is the least number of bytes the step must read from
            global memory for that row?
          </>
        }
      >
        {(c) => (
          <div data-reveal="bytes">
            <p>
              Every token contributes 4 bytes of logit and 4 bytes of count, so the floor is {n(VOCAB.qwen3)} × 8 ={' '}
              <b>{n(one)} bytes</b> per row
              {c === one ? ', exactly your number' : Math.abs(c - one) / one < 0.05 ? ', close to your number' : ''}. At batch
              64 that is {n(one * 64)} bytes, {(one * 64 / 1e6).toFixed(1)} MB, read for one token per request.
            </p>
            <p>
              Against that, each stage does a handful of operations per token: a compare, a multiply, a subtraction, an
              exponential. None comes close to the arithmetic per byte at which a GPU stops waiting for memory (the
              roofline of <a href={href('ch05', '5.1')}>section 5.1</a>; applying it to sampling is ours, and your own
              bandwidth comes from lab 5.1, not from this page). Every stage is memory-bound, and the questions worth
              asking are how many times the row is read and written, and whether the device is busy at all.
            </p>
          </div>
        )}
      </Predict>
      <p>
        That second question has an awkward answer. Each kernel in this module gives one block to one row: the shape of
        lab 10.3's softmax, of FlashInfer's sampling kernel and of llama.cpp's Metal top-k. At batch 1 that is one block on
        one multiprocessor, and the rest of the GPU idles, as <a href={href('ch10', '10.10')}>section 10.10</a> showed for
        the softmax. How much that costs is something you will measure in lab X1.3, which times a sampling step at
        batch 1 and batch 64 and against one CPU thread.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${FACTS}, A6; notes/today/ch12-facts.md D2)`}>
        <p>
          FlashInfer's <code>SamplingFromProbKernel</code>: one block per row, <code>curand_init(philox_seed, bx, ...)</code>.
          llama.cpp's Metal top-k: one threadgroup per row.
        </p>
      </Today>

      <h2>What "same seed, same token" will mean</h2>
      <p>
        By the end of the module, M5 promises two things for a given seed. First, the same random bits on every backend,
        from a counter-based generator keyed by the seed, the row and the decode step (X1.7). Second, the same token,
        except when the random number lands within a stated margin of a boundary between two tokens' shares of the
        probability (X1.8). The second promise is weaker on purpose.
      </p>
      <p>
        The reasons are facts you can check. cuRAND and hipRAND on AMD turn the same 32 bits into different floats for
        about 0.4 % of inputs. Metal's shading language has no random number function at all. And Metal may round float
        arithmetic toward zero, so a cumulative sum need not match the CPU's in the last bit.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${FACTS}, B2, C1, C5)`}>
        <p>
          <code>curand_uniform</code>: x·2⁻³² + 2⁻³³; rocRAND on AMD: 2⁻³² + x·2⁻³². MSL 4.1: no random, noise or hash
          function. MSL §8.2: "Either round ties to even or round toward zero rounding mode may be supported".
        </p>
      </Today>

      <h2>Before you start</h2>
      <p>
        Six questions on what this module assumes from chapters 9 to 14 and their labs. Each links to the section it comes
        from, and each comes back in <a href={href('review')}>review</a>.
      </p>
      <Ask q={q['x1/start/q-max-subtract']} label="From chapter 10" />
      <Ask q={q['x1/start/q-exclusive']} label="From chapter 11" />
      <Ask q={q['x1/start/q-stable']} label="From chapter 12" />
      <Ask q={q['x1/start/q-vocab-kib']} label="From chapter 9" />
      <Ask q={q['x1/start/q-msd-bin']} label="From chapter 14" />
      <Ask q={q['x1/start/q-one-block']} label="From chapters 4 and 10" />

      <h2>The labs</h2>
      <p>
        Four labs, all on the <a href={href('x1', 'labs')}>labs page</a>. Each asks for a prediction before you measure
        and tests against a CPU reference.
      </p>
      <ul>
        <li><b>Lab X1.1</b>, top-k by radix select, measured against a full sort.</li>
        <li><b>Lab X1.2</b>, top-p and min-p: composed from earlier labs, then fused into one kernel.</li>
        <li><b>Lab X1.3</b>, milestone M5: a batched GPU sampler, from logits and counts to one token per row.</li>
        <li><b>Lab X1.4</b>, filter hunt: three defects in a fused top-p and min-p kernel.</li>
      </ul>
    </>
  );
}
