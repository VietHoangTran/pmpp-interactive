import { Ask, type Question } from '../../components/Ask';
import { Beyond, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';

// The opening bridge (notes/today/ch09-facts.md D5) and the prerequisite check, drawn
// from the notes of chapters 4, 5 and 6.
export const questionsS90: Record<string, Question> = {
  'ch09/start/q-mlx': {
    kind: 'choice',
    id: 'ch09/start/q-mlx',
    prompt: (
      <>
        <pre
          className="mono"
          style={{ margin: '0 0 0.6rem', padding: '0.5rem 0.7rem', background: 'var(--cell)', border: '1px solid var(--rule)', borderRadius: 3, overflowX: 'auto' }}
        >
          {`import mlx.core as mx
a = mx.array([0, 0])
idx = mx.array([0, 1, 0, 1])
a[idx] += 1`}
        </pre>
        What does <code>a</code> hold afterwards?
      </>
    ),
    options: [
      { label: '[2, 2]', why: 'That is what a.at[idx].add(1) gives. The plain += does not apply every update.' },
      { label: '[1, 1]', correct: true },
      { label: 'An error, because the indices repeat.', why: 'MLX accepts repeated indices and documents what this line does.' },
      { label: 'It varies from run to run.', why: 'MLX\'s documentation shows one definite result for exactly this example.' },
    ],
    explain: (
      <>
        <p>
          The line is three steps: a gather, an add and a scatter. <code>a[idx]</code> reads <code>[0, 0, 0, 0]</code>,
          adding 1 gives <code>[1, 1, 1, 1]</code>, and the scatter writes 1 into <code>a[0]</code> twice and into{' '}
          <code>a[1]</code> twice. Both writes to an index carry the same old value plus one, so one increment per index
          is lost.
        </p>
        <p>
          MLX documents exactly this result and offers <code>a.at[idx].add(1)</code>, which applies every update and
          gives <code>[2, 2]</code>. PyTorch says the same thing more bluntly: with duplicate indices and{' '}
          <code>accumulate=False</code>, <code>index_put_</code> is undefined.
        </p>
      </>
    ),
    source: 'notes/today/ch09-facts.md D5 (MLX docs for array.at; PyTorch 2.15 docs for Tensor.index_put_)',
  },
  'ch09/start/q-barrier-scope': {
    kind: 'choice',
    id: 'ch09/start/q-barrier-scope',
    prompt: (
      <>
        Every block of a kernel adds its results into one array in global memory and then calls{' '}
        <code>__syncthreads()</code>. After its barrier, can block 1 rely on seeing block 0's additions?
      </>
    ),
    options: [
      { label: 'Yes: the barrier waits for every thread that touches the array, whichever block it is in.', why: 'A barrier waits for the threads of its own block, and only those.' },
      { label: 'Yes, because block 0 has the lower index, and blocks start and finish in index order.', why: 'No order between blocks is promised. Block 1 may finish before block 0 starts.' },
      { label: 'No: the barrier covers one block\'s threads, and blocks run in any order, not all at once.', correct: true },
    ],
    explain: (
      <p>
        A barrier is a meeting point for the threads of one block (<a href={href('ch04', '4.3')}>section 4.3</a>). The
        runtime assigns blocks to the hardware in no promised order, and a large grid is never all on the device at once.
        So results from different blocks cannot be put in order with barriers. This chapter merges them with atomics,
        which need no order at all.
      </p>
    ),
    source: 'ch. 4 §4.3, pp. 70–74',
  },
  'ch09/start/q-shared-init': {
    kind: 'choice',
    id: 'ch09/start/q-shared-init',
    prompt: (
      <>
        A kernel declares <code>__shared__ unsigned int counts[256];</code> and its threads immediately start
        incrementing entries. What did the counts start at?
      </>
    ),
    options: [
      { label: 'Zero: shared arrays are cleared when a block starts.', why: 'Nothing clears them. The course\'s emulator fills shared memory with 0xCD bytes so that this mistake shows every time.' },
      { label: 'Whatever that memory held: nobody initialised this block\'s copy.', correct: true },
      { label: 'The counts the previous block on the same SM left behind.', why: 'Maybe, maybe not. Nothing promises any particular content, so treat it as unknown.' },
    ],
    explain: (
      <p>
        Each block gets its own copy of a shared variable, alive while the kernel runs, and no initial value (
        <a href={href('ch05', '5.2')}>section 5.2</a>). The private histograms of section 9.4 therefore clear their
        copy first, with the whole block helping, and put a barrier after the clearing: without it a fast thread could
        count into an entry that a slow thread has not cleared yet.
      </p>
    ),
    source: 'ch. 5 §5.2, pp. 98–104; ch. 9 Fig. 9.10, p. 214; the emulator\'s 0xCD fill: labs/README.md',
  },
  'ch09/start/q-latency': {
    kind: 'choice',
    id: 'ch09/start/q-latency',
    prompt: (
      <>
        A DRAM access takes hundreds of clock cycles, yet a vector-addition kernel can run close to the memory's peak
        bandwidth. Why?
      </>
    ),
    options: [
      { label: 'Caches hide the latency of every access after the first, so only one load pays it.', why: 'Vector addition reads each element once. There is nothing for a cache to reuse.' },
      { label: 'Many warps keep independent requests in flight, so their latencies overlap.', correct: true },
      { label: 'The compiler reorders the loads so that each one is issued early and takes less time.', why: 'Reordering does not shorten an access. Overlapping many of them is what helps.' },
    ],
    explain: (
      <p>
        While one warp waits for its data, the SM runs another (<a href={href('ch04', '4.6')}>section 4.6</a>), and the
        DRAM's channels and banks work on many requests at the same time (<a href={href('ch06', '6.2')}>section 6.2</a>).
        That only works when the requests are independent. Section 9.3 is about the case where they are not: updates to
        one location have to take turns.
      </p>
    ),
    source: 'ch. 4 §4.6, pp. 82–85; ch. 6 §6.2, pp. 132–137',
  },
  'ch09/start/q-coalesce': {
    kind: 'choice',
    id: 'ch09/start/q-coalesce',
    prompt: (
      <>
        Each thread reads one byte of <code>image</code> in every step <code>k</code> of a loop. With which index do the
        loads of a warp in one step fall on consecutive bytes?
      </>
    ),
    options: [
      { label: <code>image[(blockIdx.x*blockDim.x + threadIdx.x) + k*blockDim.x*gridDim.x]</code>, correct: true },
      {
        label: <code>image[(blockIdx.x*blockDim.x + threadIdx.x)*K + k]</code>,
        why: 'Here neighbouring threads are K bytes apart in every step, so the warp\'s loads are spread over K times as much memory.',
      },
      { label: 'Both: the hardware merges any loads a warp makes in the same step.', why: 'It merges loads that fall in the same pieces of memory. Spread-out loads need more pieces.' },
    ],
    explain: (
      <p>
        In the first, thread <i>t</i> + 1 reads the byte after thread <i>t</i> in every step, so the warp's loads merge
        into few memory transactions (<a href={href('ch06', '6.1')}>section 6.1</a>). The second gives each thread its
        own contiguous run of <code>K</code> bytes, a good layout for a CPU thread and a bad one for a warp. Section 9.5
        compares exactly these two index expressions.
      </p>
    ),
    source: 'ch. 6 §6.1, pp. 124–132',
  },
  'ch09/start/q-coarsen': {
    kind: 'choice',
    id: 'ch09/start/q-coarsen',
    prompt: <>When does giving each thread several elements (thread coarsening) make a kernel faster?</>,
    options: [
      { label: 'Fewer threads means less scheduling work for the hardware, so it helps whatever the kernel does.', why: 'With fully independent threads and no such cost there is little to gain, and too few blocks leave part of the device idle.' },
      { label: 'When a per-block cost is paid for blocks the hardware would have run one after another anyway.', correct: true },
      { label: 'When the kernel is compute-bound, since coarsening raises the arithmetic each thread does.', why: 'The test is whether parallelising costs something that coarsening removes, not the kernel\'s arithmetic intensity.' },
    ],
    explain: (
      <p>
        Coarsening pays only when there is an overhead to cut (<a href={href('ch06', '6.5')}>section 6.5</a>). Section
        9.5 finds one: each private histogram of section 9.4 has to be cleared and merged once per block, so fewer,
        busier blocks do less of that work.
      </p>
    ),
    source: 'ch. 6 §6.5, pp. 141–143',
  },
};

const ROADMAP: { sec: string; title: string; does: string }[] = [
  { sec: '9.2', title: 'Atomic operations', does: 'the correct baseline: every update to a bin made indivisible' },
  { sec: '9.3', title: 'Latency and throughput of atomics', does: 'why that baseline crawls when many threads hit one bin' },
  { sec: '9.4', title: 'Privatisation', does: 'a private copy of the bins per block, merged at the end' },
  { sec: '9.5', title: 'Thread coarsening', does: 'fewer blocks, so fewer copies to clear and merge' },
  { sec: '9.6', title: 'Thread-level privatisation', does: 'a private count in each thread for a run of equal values' },
];

export function S90() {
  const q = questionsS90;
  return (
    <>
      <p>
        Start with a line of Python you could have written last week. Answer before reading on.
      </p>
      <Ask q={q['ch09/start/q-mlx']} label="Predict" />
      <p>
        The line looks like one operation and is three: read the selected entries, add one, write them back. When an
        index appears twice, two writes carry the same stale value and one increment disappears. Nothing failed; the
        library did what the statement says. A GPU kernel in which many threads update a shared array has the same
        shape, at a much larger scale and without a library to pin down the result. This chapter is that bug on the
        GPU, and its cure.
      </p>
      <Today checked="2026-11-20 (notes/today/ch09-facts.md, D5)">
        <p>
          MLX's documentation for <code>array.at</code> uses this exact example. mlx-lm uses both forms on purpose: a
          plain subtraction for the presence penalty, where duplicates should collapse, and{' '}
          <code>.at[...].subtract</code> for the frequency penalty, where every occurrence counts.
        </p>
      </Today>

      <h2>What is new in this chapter</h2>
      <p>
        Every kernel so far obeyed one rule: each output element belongs to one thread, which computes it and writes
        it. The book calls this owner-computes. Threads never wrote to the same place, so they never had to coordinate.
        A histogram breaks the rule. Which bin a thread updates depends on the pixel it reads, so any thread may update
        any bin, and several may update the same one at the same moment. The book calls the result output interference.
      </p>
      <Source pages="201" />
      <p>The chapter builds a correct kernel first and then makes it fast, in five steps:</p>
      <table className="data" data-part="roadmap">
        <thead>
          <tr>
            <th>Section</th>
            <th>What it adds</th>
          </tr>
        </thead>
        <tbody>
          {ROADMAP.map((r) => (
            <tr key={r.sec}>
              <td>
                <a href={href('ch09', r.sec)}>
                  {r.sec} {r.title}
                </a>
              </td>
              <td>{r.does}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>
        The book insists that whether each optimisation pays off depends on the hardware and on the input data. A bin
        that every thread wants behaves very differently from bins that are hit evenly. That is why every lab in this
        chapter measures two kinds of input: uniform bytes and a skewed image.
      </p>
      <Source pages="201" />

      <h2>Where this sits in LLM inference</h2>
      <p>
        Matrix multiplication and attention are owner-computes: every output element has one writer. Data-dependent
        writes appear around them. Repetition penalties need a count of how often each token id has appeared, and
        counting token ids is a histogram whose bins are the vocabulary (milestone M5, <a href={href('ch09', 'labs')}>lab 9.3</a>).
        Bucketed top-k selection counts the logits into buckets first. And any scatter-add, an output indexed by data,
        has the same race and takes the same cure.
      </p>
      <Beyond>
        <p>
          The book makes no link to language models in this chapter. The mapping is ours (notes/ch09.md). The sections
          and the summary page name the real code that does these things today (notes/today/ch09-facts.md D1 to D6).
        </p>
      </Beyond>
      <p>
        What transfers most is not the histogram itself. It is the ability to estimate how fast atomic updates can go,
        and the privatise-then-commit shape of the fast kernels. Chapter 10 reuses both: its multi-block reduction lets
        each block finish privately and commit with one atomic (Fig. 10.18, pp. 244–245).
      </p>

      <h2>Check what this chapter assumes</h2>
      <p>
        Five questions on what this chapter assumes from chapters 4 to 6. Each explanation links back to where it was
        taught, and each comes back in <a href={href('review')}>review</a>.
      </p>
      <Ask q={q['ch09/start/q-barrier-scope']} label="From chapter 4" />
      <Ask q={q['ch09/start/q-shared-init']} label="From chapter 5" />
      <Ask q={q['ch09/start/q-latency']} label="From chapters 4 and 6" />
      <Ask q={q['ch09/start/q-coalesce']} label="From chapter 6" />
      <Ask q={q['ch09/start/q-coarsen']} label="From chapter 6" />

      <h2>Your main witness: the race backend</h2>
      <p>
        Lab 4.3 introduced the emulator's second flavour, <code>race</code>. In this chapter it becomes the main witness.
        The ordinary emulator, <code>emu</code>, runs one thread at a time and switches only at barriers, so a thread's
        read, add and write are never split: a missing atomic passes there every time. The <code>race</code> backend runs
        the threads concurrently under ThreadSanitizer and names both source lines of every unsynchronised pair of
        accesses, even in a run that happened to produce the right counts. Use{' '}
        <code>python3 labs/pmpp.py run 9.1 --backend race</code> before you trust a histogram kernel.
      </p>
      <Beyond>
        <p>
          Both emulators are this course's tools, not the book's. What each one can and cannot see is listed in
          labs/README.md.
        </p>
      </Beyond>
    </>
  );
}
