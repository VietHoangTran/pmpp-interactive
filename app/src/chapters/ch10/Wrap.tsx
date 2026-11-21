import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Beyond, Source } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { functionSource } from '../../sim/toolchain';
import { STATUS } from './status';

// The kernel for the first question is a snippet of our own, loaded at run time, so there
// is no copy to drift: labs/ch10/snippets/wrap_centre.cu. Wrap.snippets.test.ts ties its
// helpers to lab 10.1, runs it on the course's emulator, and pins the numbers below.
export const CENTRE_FILE = 'ch10/snippets/wrap_centre.cu';
export const CENTRE = { block: 256, cols: 1000 } as const;
/** The columns of a row that thread 0 writes in centreRows: 0, block, 2 block, ... */
export const threadZeroColumns = (block: number, cols: number): number[] =>
  Array.from({ length: Math.ceil(cols / block) }, (_, k) => k * block).filter((c) => c < cols);

/**
 * Fig. 10.10's tree on a block of any size: thread t first adds inputs t and t + B, then
 * for stride = B/2, B/4, ... down to 1 (integer halving) thread t < stride adds slot
 * t + stride into slot t. Returns how many of the 2B inputs slot 0 ends up holding.
 */
export function sharedTreeCovers(block: number): number {
  const slots = Array.from({ length: block }, () => 2);
  for (let s = Math.floor(block / 2); s >= 1; s = Math.floor(s / 2)) for (let t = 0; t < s; t++) slots[t] += slots[t + s];
  return slots[0];
}
export const ODD_BLOCK = 96;

export const questionsWrap: Record<string, Question> = {
  'ch10/wrap/q-centre': {
    kind: 'choice',
    id: 'ch10/wrap/q-centre',
    prompt: (
      <>
        The kernel above centres each row of a <code>[rows × {CENTRE.cols}]</code> matrix, <code>y = x − mean(x)</code>,
        the first step of LayerNorm, with one block of {CENTRE.block} threads per row. It sums the row with the two-stage
        warp reduction of section 10.8, then every thread divides the <code>s</code> it holds by the row length and
        writes its columns. It compiles and runs. Which outputs are right?
      </>
    ),
    options: [
      {
        label: 'All of them: the barrier before stage 2 makes the total visible to every thread.',
        why: 'That barrier orders the hand-over of the warp sums. The total is computed after it, in warp 0\'s registers, and nothing hands it back.',
      },
      {
        label: `Only the columns that thread 0 writes: ${threadZeroColumns(CENTRE.block, CENTRE.cols).join(', ')} of each row.`,
        correct: true,
      },
      {
        label: 'The columns warp 0 writes: every lane of warp 0 ran the second warp_reduce.',
        why: 'A shuffle-down tree leaves the sum of all its inputs in lane 0 only. Lane 1 ends with lanes 1 to 31\'s share, lane 2 with less, and so on.',
      },
      {
        label: 'None: stage 2 reads partial_s before every warp has written its sum.',
        why: 'The __syncthreads() between the lane-0 writes and warp 0\'s reads is there.',
      },
    ],
    explain: (
      <>
        <p>
          <code>warp_reduce</code> built on shuffle-down leaves the sum in lane 0 only (p. 240), and stage 2 runs only in
          warp 0, so when it ends the row's total exists in one register: thread 0's. Every other thread divides what it
          last held, a partial sum of stage 2 in warp 0 and its own warp's share or less elsewhere. Thread 0 writes
          columns {threadZeroColumns(CENTRE.block, CENTRE.cols).join(', ')}; the other{' '}
          {CENTRE.cols - threadZeroColumns(CENTRE.block, CENTRE.cols).length} of each row are wrong.
        </p>
        <p>
          A block's result has to be handed back the way the warp sums were handed over: thread 0 writes it to shared
          memory, a barrier, then every thread reads it. A softmax needs the same for its max and for its sum, which is
          why lab 10.3's kernels do it. Metal's <code>simd_sum</code> gives the sum to every lane of one SIMD-group, not
          to the other SIMD-groups, so the hand-back is needed there too.
        </p>
      </>
    ),
    source: 'ours, from ch. 10 §10.7 p. 240 (only lane 0\'s value is the sum) and §10.8 pp. 241–243; run on the course emulator by Wrap.snippets.test.ts; notes/today/ch10-facts.md C1',
  },
  'ch10/wrap/q-oddblock': {
    kind: 'numeric',
    id: 'ch10/wrap/q-oddblock',
    prompt: (
      <>
        Fig. 10.10, the shared-memory kernel, is compiled and launched with <code>BLOCK_DIM = blockDim.x = {ODD_BLOCK}</code>{' '}
        (three warps) on a segment of {2 * ODD_BLOCK} values. Its loop halves the stride with integer division. How many
        of the {2 * ODD_BLOCK} values does <code>*output</code> include?
      </>
    ),
    answer: sharedTreeCovers(ODD_BLOCK),
    unit: 'values',
    explain: (
      <p>
        The load leaves {ODD_BLOCK} slots of 2 values each. Strides 48, 24, 12 and 6 halve cleanly, leaving slots 0 to 5
        with 32 values each. At stride 3, threads 0 to 2 add slots 3 to 5: three slots of 64. The next stride is 3 / 2 =
        1, so thread 0 adds slot 1 and slot 2 is never read again. The result holds {sharedTreeCovers(ODD_BLOCK)} of the{' '}
        {2 * ODD_BLOCK} values, and no barrier or bounds check can notice. Every odd stride above 1 strands a slot. The
        fixes are a power-of-two block, which is what lab 10.1's launchers insist on, or padding the shared array to the
        next power of two with the identity.
      </p>
    ),
    source: 'our analysis of Fig. 10.10, p. 237; the book assumes power-of-two blocks without saying so (notes, "Unclear in the PDF")',
  },
  'ch10/wrap/q-columns': {
    kind: 'choice',
    id: 'ch10/wrap/q-columns',
    prompt: (
      <>
        You need the sum over the batch of a row-major <code>[64 × 4096]</code> activation: one result per column, 4,096
        sums of 64 values each, the shape of a mean over tokens or a bias gradient. Which kernel?
      </>
    ),
    options: [
      {
        label: 'One block of 64 threads per column, then the shared-memory tree of section 10.6.',
        why: 'Each block\'s 64 loads are 4,096 floats apart, so no two share a memory request, and the tree spends barriers on 64 values.',
      },
      {
        label: 'One thread per column, each adding its column\'s 64 values in a loop down the rows.',
        correct: true,
      },
      {
        label: 'One thread per element, each adding its value into its column\'s result with an atomic.',
        why: '262,144 atomics, 64 on every result, landing in no fixed order: slower than a loop, and the float sums change from run to run.',
      },
      {
        label: 'Transpose the matrix first, then give each of the 4,096 rows one block, as lab 10.3 does.',
        why: 'The transpose reads and writes the whole matrix once more to reach a layout the loop does not need: neighbouring threads already read neighbouring addresses.',
      },
    ],
    explain: (
      <p>
        There are already 4,096 independent reductions, 4,096 threads of work before any tree. With one thread per column,
        the 32 threads of a warp read 32 neighbouring floats of one row at every step of their loops: coalesced, as
        section 10.5 wants. Each loop is section 10.10's coarsening taken to its end, with no barrier, no shared memory,
        no idle lanes and a fixed order of additions. A tree pays when there are few long reductions; here there are many
        short ones, and the reduction runs along the dimension that is not contiguous.
      </p>
    ),
    source: 'ours, from ch. 10 §10.5 pp. 234–235 and §10.10 pp. 246–248',
  },
  'ch10/wrap/q-bits': {
    kind: 'choice',
    id: 'ch10/wrap/q-bits',
    prompt: (
      <>
        Four reductions over the same large array, each with the book's multi-block design: a tree per block, then one
        atomic commit per block. (1) A float sum. (2) A float max, committed with a compare-and-swap loop. (3) An integer
        count of tokens that cannot overflow. (4) The largest |x|, for a quantisation scale. You run each twice on the
        same input. Which can give different bits the second time?
      </>
    ),
    options: [
      {
        label: 'All four: the blocks commit in a different order on every run.',
        why: 'A different order changes the result only when the operator rounds or depends on order.',
      },
      { label: 'Only (1): the float sum rounds after every commit.', correct: true },
      {
        label: '(1) and (2): both are floating point.',
        why: 'A max selects one of its inputs. It never rounds, so any order selects the same one (for finite values).',
      },
      {
        label: 'None: each atomic commit is exact.',
        why: 'Each commit is indivisible, yes. What changes between runs is the order of the commits, and a float sum rounds after each one.',
      },
    ],
    explain: (
      <p>
        An atomic gives each commit to itself; it gives the commits no order (p. 245). So the result is repeatable exactly
        when the operator does not care about order. Integer addition is associative, as long as it cannot overflow
        (footnote 2, p. 224). Max and the max of |x| select an input, so there is nothing to round (NaN aside, footnote 3,
        p. 225). Float addition rounds after every step, so another order can end in other last bits, which is what lab
        10.1's determinism count measures on your GPU.
      </p>
    ),
    source: 'ch. 10 §10.2 pp. 224–225 and §10.9 p. 245; the four cases are ours',
  },
  'ch10/wrap/q-greedy': {
    kind: 'choice',
    id: 'ch10/wrap/q-greedy',
    prompt: (
      <>
        In greedy mode a sampler computes <code>probs = softmax(logits / T)</code> with T &gt; 0, then returns{' '}
        <code>argmax(probs)</code>. Which reductions over the row does it actually need?
      </>
    ),
    options: [
      {
        label: 'A max, a sum, then an argmax: three passes over the row.',
        why: 'That computes the probabilities, which greedy decoding never uses.',
      },
      { label: 'A single argmax over the raw logits, with its tie rule.', correct: true },
      {
        label: 'A max over the logits: the largest logit is the token.',
        why: 'The max gives the value; the token is its index. That takes an argmax, a reduction of (value, index) pairs.',
      },
      {
        label: 'A sum: the normaliser decides which probability is largest.',
        why: 'Every probability in the row is divided by the same sum, which changes no order.',
      },
    ],
    explain: (
      <p>
        Dividing by T &gt; 0, subtracting the max, taking <code>exp</code> and dividing by one shared sum all keep the
        order of the entries, so they never change which one is largest. Greedy decoding is an argmax over the logits:
        one reduction of (value, index) pairs with a tie rule, which is lab 10.3's argmax and what its decoder calls.
        Computing the probabilities first is worse than wasted work: rounding in <code>exp</code> and in the division
        can turn two close logits into equal probabilities, and then the tie rule picks a token the logits did not tie
        on.
      </p>
    ),
    source: 'ours; the tie rule from ch. 10 §10.2 p. 225 and notes/today/ch10-facts.md D4',
  },
};

/** The kernel of the first question, from the snippet file. */
function CentreKernel() {
  const file = useLabFile(CENTRE_FILE);
  const fn = file && !file.startsWith('// could not load') ? functionSource(file, 'centreRows') : null;
  return (
    <Code
      tabs={[
        {
          label: 'CUDA (not a lab file: for this question only)',
          code: fn ? fn.text : file ?? `// loading labs/${CENTRE_FILE}`,
          status: STATUS.wrong,
          tiers: { x: 'global', y: 'global', row: 'global', out: 'global', partial_s: 'shared', s: 'reg', mean: 'reg', c: 'reg' },
        },
      ]}
      caption={
        <>
          From <code>labs/{CENTRE_FILE}</code>. Its mask type, <code>warpIdx</code>, <code>laneIdx</code> and{' '}
          <code>warp_reduce</code> are lab 10.1's. The unit tests run it on the course's emulator.
        </>
      }
    />
  );
}

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const m4 = backends.filter((b) => labStatus(reports[b], 'ch10/lab3_softmax_argmax', 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch10/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul>
        <li>
          A reduction is one loop over a binary operator, started at its identity. A tree gives the loop's answer only if
          the operator is associative, and any order of operands only if it is also commutative. Float addition is
          commutative but, once rounded, not associative, so a different tree, block count or commit order can change
          the last bits.
        </li>
        <li>
          Start a max at −FLT_MAX or at the first element, not at −∞ where a backend may not honour infinities. An
          argmax reduces (value, index) pairs, with a tie rule that makes the combine commutative.
        </li>
        <li>
          Where the active threads sit decides the cost. Keep them together, so that whole warps fall idle at once and a
          warp's accesses are neighbours.
        </li>
        <li>
          Keep partial results close: shared memory before global memory, registers and shuffles before shared memory,
          one barrier per hand-off. Once the data is loaded, a reduction is waiting more than computing.
        </li>
        <li>
          Warp code that ports: a typed lane mask, the width read at run time, the stage-2 read guarded, and no lane
          leaving before the shuffles. A shuffle-down tree leaves the result in lane 0 only.
        </li>
        <li>
          Across blocks an atomic orders nothing; a second pass fixes the order. Coarsen until there are just enough
          blocks to fill the device, and remember that one block per row at batch 1 is far from enough.
        </li>
      </ul>
      <Source section="10.11" pages="249" />

      <h2>A mixed check</h2>
      <p>
        These draw on the whole chapter and on situations the sections did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
      </p>
      <Beyond>
        <p>The kernel and the situations below are ours.</p>
      </Beyond>
      <CentreKernel />
      <Ask q={q['ch10/wrap/q-centre']} label="Find the defect" />
      <Ask q={q['ch10/wrap/q-oddblock']} />
      <Ask q={q['ch10/wrap/q-columns']} />
      <Ask q={q['ch10/wrap/q-bits']} />
      <Ask q={q['ch10/wrap/q-greedy']} />

      <h2>The book's exercises</h2>
      <p>
        The chapter ends with six exercises (pp. 249–250). Each is worked somewhere in this course; the book prints no
        answers, so the answers here are ours.
      </p>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }} data-table="exercises">
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
              <td>Divergent warps in the 5th iteration of the simple kernel, 1,024 values</td>
              <td>
                a question and the warp map in <a href={href('ch10', '10.4')}>10.4</a>
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td>The same for the convergent kernel</td>
              <td>
                a question in <a href={href('ch10', '10.4')}>10.4</a>
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td>The mirrored kernel, with the total in the last element</td>
              <td>
                "Mirror it" and a question in <a href={href('ch10', '10.4')}>10.4</a>
              </td>
            </tr>
            <tr>
              <td>4</td>
              <td>Fig. 10.20 as a max reduction</td>
              <td>
                the commit table and two questions in <a href={href('ch10', '10.9')}>10.9</a>, the text of{' '}
                <a href={href('ch10', '10.10')}>10.10</a>, and the row max of <a href={href('ch10', 'labs')}>lab 10.3</a>
              </td>
            </tr>
            <tr>
              <td>5</td>
              <td>Fig. 10.20 for any length</td>
              <td>
                the segment calculator in <a href={href('ch10', '10.9')}>10.9</a>, and kernel (c) of{' '}
                <a href={href('ch10', 'labs')}>lab 10.1</a>
              </td>
            </tr>
            <tr>
              <td>6a</td>
              <td>Tracing {'{6, 2, 7, 4, 5, 8, 3, 1}'} through the simple kernel</td>
              <td>
                the stepper and a question in <a href={href('ch10', '10.3')}>10.3</a>
              </td>
            </tr>
            <tr>
              <td>6b</td>
              <td>The same through the convergent kernel</td>
              <td>
                a question in <a href={href('ch10', '10.4')}>10.4</a>, and "Mirror it" with the exercise's data
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Where the PDF looks wrong</h2>
      <p>
        The copy of the book this course was built from has a number of apparent slips in chapter 10, most of them in
        line and figure references, and a few listings that are narrower than they look. They are listed so that you are
        not thrown by them when you read the book alongside. Where one changes code, the course's labs and snippets have
        a corrected, tested version.
      </p>
      <ul data-list="errata">
        <li>
          Fig. 10.10 (p. 237) writes <code>*output</code> but has no <code>output</code> parameter. Lab 10.1's kernel (a)
          restores it.
        </li>
        <li>
          Figs. 10.16 and 10.18 let every lane of warp 0 read <code>partialSums_s[t]</code>, which has{' '}
          <code>BLOCK_DIM / WARP_SIZE</code> entries: right only for 1,024 threads in 32-wide warps, and never on 64-wide
          AMD warps (<a href={href('ch10', '10.8')}>10.8</a>). The labs guard the read and pad with the identity.
        </li>
        <li>
          Fig. 10.18 never uses its <code>N</code>, and nothing shows <code>*output</code> being set to 0 before it runs;
          the text calls it a few changes from Fig. 10.16, though it also writes the lane and warp helpers out inline.
          Fig. 10.20 has no return type and stops at an ellipsis. Lab 10.1's kernel (c) is the whole thing.
        </li>
        <li>
          Fig. 10.10 mixes the compile-time <code>BLOCK_DIM</code> with the launched <code>blockDim.x</code>, which must
          agree, and assumes a power-of-two block (<a href={href('ch10', '10.6')}>10.6</a>). Fig. 10.11 does not say what
          a lane receives when its source lane is past the warp or not in the mask (<a href={href('ch10', '10.7')}>10.7</a>).
          Both are our readings.
        </li>
        <li>
          Fig. 10.13 passes the literal <code>0xffffffff</code>, which HIP rejects at compile time
          (<a href={href('ch10', '10.7')}>10.7</a>; notes/today/ch10-facts.md B1). It is CUDA-only, not wrong for CUDA.
        </li>
        <li>
          p. 234 calls 384 "almost half of 736", the resources of the simple kernel. p. 231 computes 864, and so does the
          course's count (<a href={href('ch10', '10.4')}>10.4</a>). The same page says the divergent iterations fall from
          ten to five; for 256 values the count is seven to five. p. 237's 36 requests for the convergent kernel come
          without a derivation; they match 12 warp-iterations of 3 accesses each (<a href={href('ch10', '10.5')}>10.5</a>).
        </li>
        <li>
          p. 229 says the stride doubles until it exceeds <code>blockIdx.x</code>; the code uses <code>blockDim.x</code>.
        </li>
        <li>
          p. 228 promises to lift the one-block limit in section 10.10; section 10.9 does it.
        </li>
        <li>
          p. 222 gives the identity of "the minimum (minimum)" operator; the second is the maximum. The same page says
          the sum is 34.6 "after iteration 5" of a five-element loop counted from 0.
        </li>
        <li>
          The sidebar on p. 227 gives its 8-team tournament "a span of four steps"; three rounds is three steps.
        </li>
        <li>
          Figure and line references that point one off: "Fig. 10.7" for the kernel of Fig. 10.8 (p. 235), "Fig. 10.9"
          for the kernel of Fig. 10.10 (p. 237), "line 06" for line 05 of Fig. 10.8 (p. 233), "line 04" for line 05 of
          Fig. 10.10 (p. 236), and "lines 17–19" for lines 15–17 of Fig. 10.14 (p. 241). p. 236 also calls the final write
          "the final write to input[0]"; the code writes <code>*output</code>. p. 234 credits coalescing to chapter 5;
          this edition teaches it in chapter 6 (<a href={href('ch06', '6.1')}>6.1</a>).
        </li>
        <li>
          Iterations are numbered from 0 in Figs. 10.6 and 10.7 and from 1 in the text of pp. 230–234. The course says
          which it means each time.
        </li>
        <li>
          p. 243 says the two-stage kernel has "more control divergence". <code>warp_reduce</code> has no branch: what it
          has more of is lanes computing sums nobody reads (<a href={href('ch10', '10.8')}>10.8</a>).
        </li>
        <li>
          The book's links for Thrust and CUB (p. 250) no longer resolve; the pages now live under{' '}
          <code>nvidia.github.io/cccl/unstable/</code> (ch10-facts A12).
        </li>
      </ul>
      <Aside>
        <p>
          These are our readings of one PDF, checked against current vendor documentation on 2026-11-20
          (notes/today/ch10-facts.md). If your copy differs, trust your copy.
        </p>
      </Aside>
      <p>
        Some of the book is left out on purpose, so that you can tell an omission from a gap. The 2010 World Cup venues
        and dates of the sidebar (pp. 226–227) are not repeated; the knockout-as-max-tree idea is in{' '}
        <a href={href('ch10', '10.2')}>10.2</a>. Every kernel here is float32, though p. 221 names other types; half and
        8-bit reductions come back in module X2, and Metal has no <code>double</code>. Thrust and CUB are named, not
        used, and exercise 4's cross-block float max is discussed, not built, because no float atomic max is portable.
        The warp vote functions of p. 241 are worked in chapter 12 (<a href={href('ch12', '12.3')}>12.3</a>). The
        single-pass (online) softmax is part of chapter 20's flash attention (<a href={href('ch20', '20.5')}>20.5</a>).
      </p>

      <h2>The project so far</h2>
      <p>
        Lab 10.3's row-wise max, softmax and argmax are milestone M4 of the through-line project: greedy decoding end to
        end over <code>[batch × vocab]</code> logits, on a lookup-table model for now.{' '}
        <span data-m4={m4.join(',') || 'none'}>
          {backends.length === 0
            ? 'No lab results are recorded on this machine yet.'
            : m4.length
              ? `Your own lab 10.3 passes on ${m4.map(where).join(' and ')} on this machine.`
              : 'Your own lab 10.3 does not pass on any backend of this machine yet.'}
        </span>{' '}
        See the <a href={href('ch10', 'labs')}>labs</a>.
      </p>
      <p>
        Chapter 11 keeps the tree and changes what it returns. A reduction gives one value; a prefix sum gives every
        partial total along the way, the running sum that a top-p sampler cuts at its probability threshold. The book
        calls reduction the foundation of that pattern (p. 249), and chapter 11 is where it meets parallel algorithms
        that do more work than the sequential loop (p. 226). Start at <a href={href('ch11')}>chapter 11</a>.
      </p>
    </>
  );
}
