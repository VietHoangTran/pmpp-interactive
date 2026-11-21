import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';

// Prerequisite questions drawn from the notes of chapters 4, 6 and 9.
export const questionsS100: Record<string, Question> = {
  'ch10/start/q-lane': {
    kind: 'numeric',
    id: 'ch10/start/q-lane',
    prompt: (
      <>
        A block of 256 threads is cut into warps of 32. Thread 75 sits in some warp at some position, its <b>lane</b>.
        What is its lane index?
      </>
    ),
    answer: 11,
    explain: (
      <p>
        Warp <i>n</i> holds threads 32<i>n</i> to 32<i>n</i> + 31, so 75 = 2 × 32 + 11: warp 2, lane 11. The lane is{' '}
        <code>threadIdx.x % 32</code> and the warp <code>threadIdx.x / 32</code>. On an AMD gfx9 device, whose warps are
        64 wide, the same thread is lane 11 of warp 1. That is why the warp-level kernels later in this chapter compute
        lane and warp from the warp width, never from a literal 32.
      </p>
    ),
    source: 'ch. 4 §4.4, p. 75; AMD warp width: notes/today/ch05-facts.md B5',
  },
  'ch10/start/q-divwarps': {
    kind: 'numeric',
    id: 'ch10/start/q-divwarps',
    prompt: (
      <>
        A 128-thread block runs <code>if (threadIdx.x &lt; 48) {'{ ... }'}</code>, with warps of 32. How many warps
        diverge at this branch?
      </>
    ),
    answer: 1,
    unit: 'warp',
    explain: (
      <p>
        Warp 0 (threads 0 to 31) takes the branch as a whole. Warp 1 (32 to 63) splits at 48: threads 32 to 47 take it,
        48 to 63 do not, so it runs both paths. Warps 2 and 3 skip it as a whole. What decides divergence is whether
        the condition differs <em>inside</em> a warp, not how many threads take the branch. Section 10.4 rearranges a
        kernel on exactly this rule.
      </p>
    ),
    source: 'ch. 4 §4.5, pp. 79–82',
  },
  'ch10/start/q-gridbarrier': {
    kind: 'choice',
    id: 'ch10/start/q-gridbarrier',
    prompt: <>Why can the blocks of an ordinary kernel launch not wait for each other at a barrier?</>,
    options: [
      {
        label: 'Blocks run in no promised order and need not all be resident, so a waiting block may never be released.',
        correct: true,
      },
      {
        label: 'Each block has its own shared memory, so blocks hold no common data that a barrier could protect.',
        why: 'Blocks could still share results through global memory. The obstacle is scheduling, not the lack of data to exchange.',
      },
      {
        label: 'A barrier across blocks would be correct, but too slow to be worth providing.',
        why: 'That is a cost argument. The problem is correctness: such a wait may never end.',
      },
      {
        label: 'Blocks run in launch order, so by the time block b runs, every lower-numbered block has finished.',
        why: 'No order is promised. Block 7 may finish before block 0 starts.',
      },
    ],
    explain: (
      <p>
        The runtime assigns whole blocks to the hardware as resources free up, in no promised order, and a large grid
        is never all resident at once. A barrier is safe only among threads that are guaranteed to be running
        together: the threads of one block. The book mentions a grid-wide barrier through cooperative groups, which
        works only under restrictions that make every block resident. Section 10.9 has to finish a sum across blocks
        without such a barrier.
      </p>
    ),
    source: 'ch. 4 §4.3, pp. 70–73',
  },
  'ch10/start/q-sectors': {
    kind: 'numeric',
    id: 'ch10/start/q-sectors',
    prompt: (
      <>
        The 32 threads of a warp execute <code>x = in[2*t];</code> with <code>t</code> = 0 to 31, on an aligned array
        of 4-byte floats. NVIDIA documents that global memory is accessed in 32-byte transactions. How many 32-byte
        pieces of memory does this one load touch?
      </>
    ),
    answer: 8,
    unit: 'transactions',
    explain: (
      <p>
        The warp reads elements 0, 2, 4, …, 62: bytes 0 to 251, which span eight 32-byte pieces. Reading{' '}
        <code>in[t]</code> instead would touch elements 0 to 31, bytes 0 to 127: four pieces. A stride of 2 doubles the
        transactions and leaves half of every byte delivered unused. Section 10.5 counts this for a reduction kernel
        whose active threads read further apart with every step.
      </p>
    ),
    source: 'ch. 6 §6.1, pp. 124–132; the 32-byte unit is NVIDIA\'s (notes/today/ch06-facts.md A2)',
  },
  'ch10/start/q-lostupdate': {
    kind: 'choice',
    id: 'ch10/start/q-lostupdate',
    prompt: (
      <>
        256 threads each execute <code>*sum += x[threadIdx.x];</code> on one <code>float</code> in global memory,
        without an atomic. What can <code>*sum</code> end up holding?
      </>
    ),
    options: [
      { label: 'The correct total: each += is a single statement, so the threads take turns.', why: 'One statement, three steps: read, add, write. Other threads can come between them.' },
      { label: 'The correct total up to rounding, because float addition is commutative and the order does not matter.', why: 'Commutativity is about the order of operands. Here whole updates disappear.' },
      {
        label: 'The sum of some of the values, from all 256 down to one: overlapping updates are lost.',
        correct: true,
      },
      { label: 'Only the last thread\'s own x, never a partial sum, because each write replaces the one before.', why: 'A thread writes old + x, so whatever value it read survives inside its write.' },
    ],
    explain: (
      <p>
        Each <code>+=</code> reads the location, adds, and writes back. If two threads read the same old value, one
        write overwrites the other and that contribution is gone. An atomic makes each read-modify-write indivisible,
        so nothing is lost. It does not fix the <em>order</em> in which the threads' updates land, which matters for a
        float sum and comes back in section 10.9.
      </p>
    ),
    source: 'ch. 9 §9.2, pp. 203–209',
  },
};

const KERNELS: { kernel: string; fig: string; sec: string; removes: string }[] = [
  { kernel: 'Simple', fig: 'Fig. 10.5', sec: '10.3', removes: 'nothing yet: the baseline, one block, in place' },
  { kernel: 'Convergent', fig: 'Fig. 10.8', sec: '10.4', removes: 'control divergence, and with it uncoalesced access (10.5)' },
  { kernel: 'Shared memory', fig: 'Fig. 10.10', sec: '10.6', removes: 'repeated trips to global memory' },
  { kernel: 'Warp shuffles at the end', fig: 'Fig. 10.14', sec: '10.7', removes: 'barriers and shared memory in the last iterations' },
  { kernel: 'Two-stage warp reduction', fig: 'Fig. 10.16', sec: '10.8', removes: 'all but one barrier and one shared-memory hand-off' },
  { kernel: 'Multi-block with an atomic', fig: 'Fig. 10.18', sec: '10.9', removes: 'the one-block limit of 2048 elements' },
  { kernel: 'Coarsened', fig: 'Fig. 10.20', sec: '10.10', removes: 'hardware sitting idle in every block\'s shrinking tail' },
];

export function S100() {
  const q = questionsS100;
  return (
    <>
      <p>
        Every decode step ends the same way. The model produces one logit per vocabulary entry, and something has to
        turn that row into a token. A numerically stable <code>torch.softmax(logits / T, -1)</code> needs the largest
        logit in the row and then a sum over the whole row. Greedy decoding is <code>torch.argmax(logits, -1)</code>.
        Inside the model, LayerNorm and RMSNorm sum along every row of activations. Each of these takes many values and
        returns one. That kind of computation is a <b>reduction</b>, and this chapter is about writing one well on a
        GPU.
      </p>
      <Beyond>
        <p>
          The book does not mention language models in this chapter. The link to softmax, argmax and normalisation
          layers is ours.
        </p>
      </Beyond>
      <p>
        The book builds all of its kernels for the plainest reduction there is, a sum, and says why. Threads in a
        reduction have to cooperate, and every way of making them cooperate has a cost. A sum is small enough that
        each cost can be exposed and removed in turn, one kernel at a time. What you learn carries over to max, argmax
        and the other reductions, under two conditions on the operator that sections{' '}
        <a href={href('ch10', '10.1')}>10.1</a> and <a href={href('ch10', '10.2')}>10.2</a> make precise.
      </p>
      <Source pages="221" />

      <h2>Seven kernels, one bottleneck each</h2>
      <p>
        After two sections on reduction itself, the chapter writes seven kernels. Each one keeps what the previous one
        got right and removes one more obstacle. Keep this table in mind as the map of the chapter.
      </p>
      <table className="data" data-part="kernel-map">
        <thead>
          <tr>
            <th>Kernel</th>
            <th>Book</th>
            <th>What it removes</th>
          </tr>
        </thead>
        <tbody>
          {KERNELS.map((k) => (
            <tr key={k.fig}>
              <td>
                <a href={href('ch10', k.sec)}>{k.kernel}</a>
              </td>
              <td className="nowrap">
                {k.fig}, §{k.sec}
              </td>
              <td>{k.removes}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>
        Most of the obstacles are familiar from chapters 4 to 6: divergent warps, scattered memory accesses, traffic to
        global memory, waiting at barriers. Two are new. Blocks cannot wait for each other, so a sum over more than one
        block needs another way to combine their results. And a tree of additions has less and less work in each step,
        so most of the hardware assigned to it is idle most of the time.
      </p>
      <Source pages="227–248" />

      <h2>Where it ends</h2>
      <p>
        Three labs follow the sections: <a href={href('ch10', 'labs')}>one sum written three ways, a hunt for broken
        reductions, and lab 10.3</a>, which is milestone M4 of the course project: a row-wise softmax and argmax over
        logits, so that greedy decoding works end to end on your own kernels.
      </p>
      <Aside>
        <p>
          <b>Colour in this course</b> means where data lives:{' '}
          <span className="chip reg">register</span> <span className="chip shared">shared memory</span>{' '}
          <span className="chip global">global memory</span> <span className="chip const">constant memory</span>. Red
          always means something failed, green that it matched.
        </p>
      </Aside>
      <p>
        The book writes CUDA only. These pages add HIP and Metal wherever they differ, and say so. The largest
        differences come in sections 10.7 and 10.8, at the warp level: HIP rejects the book's 32-bit lane mask and may
        run 64-wide warps, and Metal offers a one-call float sum across a SIMD-group that CUDA does not have.
      </p>
      <Today checked="2026-11-20 (ch10-facts A4, B1, C1)">
        <p>
          HIP's <code>_sync</code> functions require a 64-bit mask; Metal has <code>simd_sum</code> and{' '}
          <code>simd_max</code> on floats; CUDA's warp reduce functions take integers only.
        </p>
      </Today>

      <h2>Before you start</h2>
      <p>
        Five questions on what this chapter assumes from chapters 4, 6 and 9. Each answer comes back in a later section;
        each question comes back in <a href={href('review')}>review</a>.
      </p>
      <Ask q={q['ch10/start/q-lane']} label="From chapter 4" />
      <Ask q={q['ch10/start/q-divwarps']} label="From chapter 4" />
      <Ask q={q['ch10/start/q-gridbarrier']} label="From chapter 4" />
      <Ask q={q['ch10/start/q-sectors']} label="From chapter 6" />
      <Ask q={q['ch10/start/q-lostupdate']} label="From chapter 9" />
    </>
  );
}
