import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Beyond, Source } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { fmt, labStatus, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { BOOK_MEMORY, perLocationRate } from '../../sim/atomicrate';
import { functionSource } from '../../sim/toolchain';

// The kernel for the first question is a snippet of our own, loaded at run time, so there
// is no copy to drift: labs/ch09/snippets/wrap_early_exit.cu. Wrap.snippets.test.ts runs it
// on the course's emulator and compares the bins it gets wrong with earlyExitLost below.
export const EARLY_FILE = 'ch09/snippets/wrap_early_exit.cu';
export const EARLY = { n: 1000, block: 256 } as const;
/** The image of the question and of the check: pixel i has value (37 i + 11) mod 256. */
export const earlyImage = (n: number): number[] => Array.from({ length: n }, (_, i) => (37 * i + 11) % 256);

/**
 * What histogramEarlyExit loses. Thread t of a block zeroes and commits the bins b with
 * b mod blockDim = t. In the last block only `live` threads stay, so the counts of that
 * block's pixels whose bin belongs to a thread that left are never committed. Returns the
 * number of threads that stay in the last block and, per bin, how many counts go missing.
 */
export function earlyExitLost(values: number[], block: number, bins = 256): { live: number; lost: Map<number, number> } {
  const n = values.length;
  const live = n % block;
  const lost = new Map<number, number>();
  if (live === 0) return { live: block, lost };
  for (let i = n - live; i < n; i++) {
    const b = values[i];
    if (b % block >= live && b < bins) lost.set(b, (lost.get(b) ?? 0) + 1);
  }
  return { live, lost };
}

/** Mixture-of-experts routing counts: the question's numbers. */
export const MOE = { tokens: 16384, topK: 2, experts: 64, block: 256 } as const;
export const moeAtomics = () => {
  const updates = MOE.tokens * MOE.topK;
  const blocks = Math.ceil(updates / MOE.block);
  return { updates, blocks, commits: blocks * MOE.experts };
};

/** Appending candidates through one global counter, in the book's DRAM model. */
export const APPEND = { candidates: 100000 } as const;
export const appendMs = () => (APPEND.candidates / perLocationRate(BOOK_MEMORY.clockHz, BOOK_MEMORY.latencyCycles)) * 1e3;

const early = earlyExitLost(earlyImage(EARLY.n), EARLY.block);
const lostBins = [...early.lost.keys()].sort((a, b) => a - b);
const lostRange = lostBins.length ? `${lostBins[0]} to ${lostBins[lostBins.length - 1]}` : 'none';
const moe = moeAtomics();

export const questionsWrap: Record<string, Question> = {
  'ch09/wrap/q-early-exit': {
    kind: 'choice',
    id: 'ch09/wrap/q-early-exit',
    prompt: (
      <>
        The kernel above is launched with {EARLY.block} threads per block on an image of {EARLY.n.toLocaleString('en-US')}{' '}
        pixels, pixel <i>i</i> holding (37<i>i</i> + 11) mod 256, so that every bin holds 3 or 4 pixels. It compiles and
        never hangs. Which counts come out wrong?
      </>
    ),
    options: [
      {
        label: 'None: a thread that has returned does not hold the others at a barrier.',
        why: 'That part is true on NVIDIA, so the barriers work. But the threads that left also had jobs besides their pixel.',
      },
      {
        label: `Bins ${lostRange}, one count short each: in the last block, nobody zeroes or commits them.`,
        correct: true,
      },
      {
        label: `Bins 0 to ${early.live - 1}: only the ${early.live} threads left in the last block count their pixels.`,
        why: 'Each remaining thread counts its own pixel, whatever its value. What is missing is somebody to zero and commit the other bins.',
      },
      {
        label: 'All of them, at random: the barriers are no longer reached by every thread of every block.',
        why: 'NVIDIA\'s barrier waits for the threads that have not exited, so the kernel is well defined there; the damage is deterministic and confined to the last block.',
      },
    ],
    explain: (
      <>
        <p>
          The zeroing and the commit loops hand out the bins by thread index: thread <i>t</i> owns every bin{' '}
          <i>b</i> with <i>b</i> mod {EARLY.block} = <i>t</i>. The first three blocks are full. The last one has{' '}
          {EARLY.n.toLocaleString('en-US')} − {EARLY.n - early.live} = {early.live} pixels, so threads {early.live} to{' '}
          {EARLY.block - 1} return at once, and nobody zeroes or commits bins {lostRange} of that block's copy. The
          remaining threads still count pixels with those values into them, and the counts stay there. The emulator shows
          exactly these {lostBins.length} bins one short, under all three thread orders, and at 128 threads a different set.
        </p>
        <p>
          A thread with no pixel still has work: its share of the zeroing and of the commit. Fig. 9.10 therefore guards
          only the update with <code>if(i &lt; width*height)</code>, and Fig. 9.15 lets a thread whose first pixel is out
          of range skip its pixels and still join the barrier and the commit. NVIDIA's barrier does wait only for threads
          that have not exited, but the book's rule, every thread reaches every barrier, is the one that keeps you out of
          this trap.
        </p>
      </>
    ),
    source: 'ours, after ch. 9 Fig. 9.10, p. 214, and Fig. 9.15, pp. 218–219; run on the course emulator by Wrap.snippets.test.ts; notes/today/ch04-facts.md A6',
  },
  'ch09/wrap/q-float-weights': {
    kind: 'choice',
    id: 'ch09/wrap/q-float-weights',
    prompt: (
      <>
        On CUDA, <code>torch.bincount(ids, weights=w)</code> with float weights raises an alert when you ask PyTorch for
        deterministic algorithms. Without weights it does not. Why?
      </>
    ),
    options: [
      {
        label: 'Float atomics can drop an update under heavy contention, which integer atomics do not.',
        why: 'Both are atomic: every update arrives. What differs is what the order of arrival does to the result.',
      },
      {
        label: 'Atomics land in no fixed order, and float sums round at every step, so the last bits can change.',
        correct: true,
      },
      {
        label: 'Weighted bins need a larger private copy, which moves the counting from shared to global memory.',
        why: 'Where the copy lives changes the speed. In either place the atomics, and the blocks\' commits, still arrive in no fixed order.',
      },
      {
        label: 'With weights, which block counts an element changes from run to run, so partial sums differ.',
        why: 'The partitioning is fixed by the indices. Every element is counted once by the same thread; only the order of the additions varies.',
      },
    ],
    explain: (
      <p>
        An atomic makes each update indivisible and orders nothing (p. 206). The book merges private copies "in any
        order" because addition is commutative and associative (p. 211). That is exact for integer counts as long as
        they cannot overflow. Float addition rounds after each step, so it is not exactly associative, and two runs can
        add the same weights in different orders. PyTorch's CUDA <code>bincount</code> raises its alert for exactly this
        reason, "because of floating point atomicAdd usage". The same holds for any float scatter-add, such as an
        embedding table's gradient.
      </p>
    ),
    source: 'ch. 9 §9.2 p. 206 and §9.4 p. 211; notes/today/ch09-facts.md D1 (PyTorch SummaryOps.cu)',
  },
  'ch09/wrap/q-moe': {
    kind: 'choice',
    id: 'ch09/wrap/q-moe',
    prompt: (
      <>
        A mixture-of-experts layer sends each of {MOE.tokens.toLocaleString('en-US')} tokens to {MOE.topK} of{' '}
        {MOE.experts} experts and needs the number of tokens each expert received. The router's choices are uneven: a few
        experts get far more than their share. Which kernel?
      </>
    ),
    options: [
      {
        label: `One thread per choice, each doing a device-scope fetch_add on one of ${MOE.experts} counters in global memory.`,
        why: `Correct but slow on exactly this input: ${moe.updates.toLocaleString('en-US')} atomics queue on a few busy counters, the skewed case of section 9.3.`,
      },
      {
        label: `A private copy of the ${MOE.experts} counters per block in shared memory, then a commit of the non-zero ones with device-scope atomics.`,
        correct: true,
      },
      {
        label: `${MOE.experts} threads, one per expert, each reading every choice and counting its own.`,
        why: `Owner-computes again, but ${MOE.experts} threads cannot fill a GPU, and each reads all ${moe.updates.toLocaleString('en-US')} choices.`,
      },
      {
        label: 'One thread per choice with a plain increment: the counts are only used to size buffers, so a few lost updates do not matter.',
        why: 'A lost update makes a count too small, and a buffer sized from it too small: tokens are written past its end.',
      },
    ],
    explain: (
      <p>
        {MOE.experts} counters of 4 bytes are 256 bytes, which fits any block's shared memory. With blocks of {MOE.block}{' '}
        threads, one choice each, the {moe.updates.toLocaleString('en-US')} updates go to {moe.blocks} private copies, and at
        most {moe.blocks} × {MOE.experts} = {moe.commits.toLocaleString('en-US')} atomics reach global memory, a quarter
        of the basic kernel's. More to the point, each busy expert's queue is split across the blocks in flight, and the
        uneven routing that hurts the basic kernel most is where privatisation pays most (sections 9.3 and 9.4). With a
        larger batch, coarsening (section 9.5) cuts the commits further.
      </p>
    ),
    source: 'ours; the design is ch. 9 §9.4, pp. 211–214, and §9.5, p. 214',
  },
  'ch09/wrap/q-narrow-counts': {
    kind: 'choice',
    id: 'ch09/wrap/q-narrow-counts',
    prompt: (
      <>
        To fit four private copies per block, a colleague changes Fig. 9.14's shared bins to{' '}
        <code>unsigned char bins_s[256]</code>, one byte per count, keeping 256 threads and a coarsening factor of 4.
        What goes wrong?
      </>
    ),
    options: [
      { label: 'Nothing: 256 one-byte bins fit easily, four copies included, in any block\'s shared memory.', why: 'Fitting was never the problem. The range of a count is.' },
      {
        label: 'A bin can receive 1,024 counts and wrap past 255, and 8-bit atomics are slow emulations or absent.',
        correct: true,
      },
      {
        label: 'Only the commit: a device-scope atomic cannot add an 8-bit value to a 32-bit public bin.',
        why: 'The commit reads the count and adds it to an unsigned int bin like any other value. The damage is done before it.',
      },
      {
        label: 'It is slower because of bank conflicts between neighbouring bytes, but every count is right.',
        why: 'A constant image puts all 1,024 of a block\'s pixels in one bin, and one byte holds at most 255.',
      },
    ],
    explain: (
      <p>
        Size a private counter for the most updates one copy can receive: here 256 threads × 4 pixels = 1,024, which needs
        more than 8 bits. Narrow atomics also cost speed, or do not exist: libcu++ says types smaller than 4 bytes "are not natively
        supported by the underlying hardware" and are emulated, "drastically slower", with contention on the surrounding
        4-byte word made worse. Metal has no 8- or 16-bit atomic type; PyTorch and MLX build them from a compare-and-swap
        loop on a 32-bit word. For a vocabulary, where a narrow counter is most tempting, the same applies.
      </p>
    ),
    source: 'ours, after ch. 9 Fig. 9.14, p. 217; notes/today/ch09-facts.md A1, C2, D7',
  },
  'ch09/wrap/q-append': {
    kind: 'numeric',
    id: 'ch09/wrap/q-append',
    prompt: (
      <>
        A sampler writes the indices of its {APPEND.candidates.toLocaleString('en-US')} surviving candidates to one array,
        each thread taking its slot with <code>pos = counter.fetch_add(1)</code> on one global counter. In the book's DRAM
        model (200-cycle latency, 1 GHz, nothing cached), how many milliseconds do those atomics take?
      </>
    ),
    answer: Number(appendMs().toFixed(3)),
    tol: 0.02,
    unit: 'ms',
    explain: (
      <p>
        One location takes one atomic per 200 + 200 cycles: 2.5 million per second. {APPEND.candidates.toLocaleString('en-US')}{' '}
        / 2.5 M = {fmt(appendMs())} ms, during which the rest of the GPU waits on one queue, however many threads you
        launch. This is the first use of <code>fetch_add</code>'s return value, which the book says chapters 12 and 18
        need (p. 209). The cure has this chapter's shape: count a block's candidates privately, then reserve the block's
        whole range with a single <code>fetch_add</code> of that count, one atomic per block. Chapter 12 builds it, in{' '}
        <a href={href('ch12', '12.4')}>section 12.4</a>.
      </p>
    ),
    source: 'the §9.3 model, pp. 209–210; p. 209 for the return value; the sampler is ours',
  },
};

/** The kernel of the first question, from the snippet file. */
function EarlyExitKernel() {
  const file = useLabFile(EARLY_FILE);
  const fn = file && !file.startsWith('// could not load') ? functionSource(file, 'histogramEarlyExit') : null;
  return (
    <Code
      tabs={[
        {
          label: 'CUDA (not a lab file: for this question only)',
          code: fn ? fn.text : file ?? `// loading labs/${EARLY_FILE}`,
          status: 'emulator-checked; wrong on purpose',
          tiers: { image: 'global', bins: 'global', bins_s: 'shared', i: 'reg', b: 'reg' },
        },
      ]}
      caption={
        <>
          From <code>labs/{EARLY_FILE}</code>: Fig. 9.10 with one change. The unit tests run it on the course's emulator.
        </>
      }
    />
  );
}

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const m5 = backends.filter((b) => labStatus(reports[b], 'ch09/lab3_tokencounts', 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch09/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <p>
        The book's summary, extended with what this course added on the backends and the tools, in the order you will
        need it.
      </p>
      <Source section="9.7" pages="219–220" />
      <ul>
        <li>
          When the data decides where a thread writes, owner-computes is gone: any thread may update any output. A plain
          update is a read, an add and a write, and two of them on one location can lose one. Formally that is a data
          race, so a lost count is the typical result, not a promised one.
        </li>
        <li>
          An atomic makes one update indivisible and orders nothing. Choose its scope from the threads that share the
          location: block for a block's private copy, device for bins that several blocks update. Pass the scope and the
          memory order every time; left out, they default to the widest and the strictest.
        </li>
        <li>
          Atomics on one location run one at a time, at about one per load-plus-store latency, and the busiest bin sets
          the pace of the whole kernel. Skewed data is the worst case, and it is the common one.
        </li>
        <li>
          Privatise: give a subset of threads its own copy and merge the copies at the end. One copy per block, in shared
          memory when it fits: zero it, barrier, count, barrier, commit the non-zero bins.
        </li>
        <li>
          Each private copy costs a zeroing and a commit. Coarsen until there are just enough blocks, and keep the
          interleaved index so a warp's reads stay neighbours. Merge runs in a register only where one thread sees long
          runs of one value.
        </li>
        <li>
          A vocabulary does not fit on chip. Count per request, in global memory, and update the counts one decode step at
          a time rather than recounting. Test every atomic on the race backend; check every scope by reading it.
        </li>
      </ul>

      <h2>A mixed check</h2>
      <p>
        These draw on the whole chapter and on situations the sections did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
      </p>
      <Beyond>
        <p>The kernel, the router, the sampler and the narrow counters are ours.</p>
      </Beyond>
      <EarlyExitKernel />
      <Ask q={q['ch09/wrap/q-early-exit']} label="Find the defect" />
      <Ask q={q['ch09/wrap/q-float-weights']} />
      <Ask q={q['ch09/wrap/q-moe']} />
      <Ask q={q['ch09/wrap/q-narrow-counts']} />
      <Ask q={q['ch09/wrap/q-append']} />

      <h2>The book's exercises</h2>
      <p>
        The chapter ends with five exercises (p. 220), the last in three parts. Each is worked somewhere in this course;
        the book prints no answers, so the answers here are ours.
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
              <td>Atomics per second on one variable at 100 ns each</td>
              <td>
                a question in <a href={href('ch09', '9.3')}>9.3</a>, and the throughput explorer there
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td>The same with atomics in L2: 4 ns on a hit, 90% hits</td>
              <td>
                a question in <a href={href('ch09', '9.3')}>9.3</a>
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td>Floating-point rate limited by exercise 1's atomics</td>
              <td>
                a question in <a href={href('ch09', '9.3')}>9.3</a>
              </td>
            </tr>
            <tr>
              <td>4</td>
              <td>The same after privatising into shared memory</td>
              <td>
                a question in <a href={href('ch09', '9.3')}>9.3</a>, with both readings of the exercise
              </td>
            </tr>
            <tr>
              <td>5a</td>
              <td>Global atomics of the basic kernel, 524,288 elements</td>
              <td>
                a question in <a href={href('ch09', '9.4')}>9.4</a>
              </td>
            </tr>
            <tr>
              <td>5b</td>
              <td>The most for shared-memory privatisation</td>
              <td>
                a question and the explorer in <a href={href('ch09', '9.4')}>9.4</a>
              </td>
            </tr>
            <tr>
              <td>5c</td>
              <td>The most with coarsening factor 4</td>
              <td>
                a question and the commit calculator in <a href={href('ch09', '9.5')}>9.5</a>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Where the PDF looks wrong</h2>
      <p>
        The copy of the book this course was built from has a few apparent slips in chapter 9, most of them left over from
        an earlier edition's text example. They are listed so that you are not thrown by them when you read the book
        alongside. Where one changes code, the labs have a corrected, tested version.
      </p>
      <Aside>
        <p>
          These are our readings of one PDF, checked against current vendor documentation on 2026-11-20
          (notes/today/ch09-facts.md). If your copy differs, trust your copy.
        </p>
      </Aside>
      <ul data-list="errata">
        <li>
          p. 211 says "In our text histogram example"; this edition's example is an image. p. 216 says "the four threads
          access characters 8 through 15", where Fig. 9.13 and the sentence before it have eight threads and pixels.
        </li>
        <li>
          p. 215 says "In Fig. 9.11, each thread block is assigned to a contiguous block segment" of{' '}
          <code>COARSE_FACTOR × blockDim.x</code> elements. That describes the code of Fig. 9.12; Fig. 9.11 is the
          drawing, whose factor of 3 the text never states.
        </li>
        <li>
          p. 213 puts the first atomic on <code>bins_s</code> at "line 14" of Fig. 9.10. The reference is built on lines
          13 and 14 and <code>fetch_add</code> is on line 15.
        </li>
        <li>
          The declaration of <code>cuda::atomic_ref</code> (p. 207) and the signature of <code>fetch_add</code> (p. 208)
          are printed without their defaults: system scope and <code>seq_cst</code>. The listings always pass both, and so
          does every kernel here (<a href={href('ch09', '9.2')}>9.2</a>).
        </li>
        <li>
          The signature on p. 208 spells the order <code>cuda::std::memory_order</code>, the listings{' '}
          <code>cuda::memory_order_relaxed</code>. Not an error: both spellings are valid.
        </li>
        <li>
          No listing zeroes <code>bins</code> or Fig. 9.9's <code>bins_pool</code>, and no host code or launch
          configuration is shown. In the labs the caller zeroes both, and the launchers are ours.
        </li>
        <li>
          Exercise 4 gives a shared-memory "access latency" of 1 ns without saying whether one atomic costs one access or
          two (a read and a write, as in the model of §9.3). <a href={href('ch09', '9.3')}>Section 9.3</a> states the
          reading it uses and gives the other.
        </li>
        <li>
          p. 213 calls shared memory "private to each SM". It sits on the SM, but each block is given, and sees, its own
          allocation. <a href={href('ch05', 'wrap')}>Chapter 5's summary</a> flags the same slip on p. 116 and in exercise 12.
        </li>
        <li>
          Fig. 9.15 declares <code>b</code> as <code>unsigned int</code> and <code>bNext</code> as{' '}
          <code>unsigned char</code>. Harmless: both hold 0 to 255, and the comparison promotes <code>bNext</code>.
        </li>
      </ul>
      <p>
        Some statements are dated rather than wrong. The 200-cycle DRAM, "tens of cycles" for the last-level cache and "a
        few cycles" for shared memory are the book's example: no vendor publishes latencies in cycles today. 1,024
        threads per block still holds. The book's third scope, "across all GPU devices", is CUDA's system scope, which
        includes the CPU. And the hint to read four pixels, a 32-bit word, per iteration (p. 216) is in no current NVIDIA
        document.
      </p>
      <Source pages="207–216" />

      <h2>What this course shortened</h2>
      <p>
        Some of the book is left out on purpose, so that you can tell an omission from a gap. The two everyday analogies
        for a lost update, the airline seat map and the store kiosks (pp. 204–205), are replaced by the interleaving
        stepper of <a href={href('ch09', '9.2')}>9.2</a>. The list of uses for histograms (pp. 202–203) is one sentence
        in <a href={href('ch09', '9.1')}>9.1</a>, and the tree image of Fig. 9.1 is replaced by our own pictures.
        Footnote 1's primer on C++ classes and templates (p. 206) is skipped. The two-copy and four-copy schemes (p. 211)
        are settings of the explorer in <a href={href('ch09', '9.4')}>9.4</a>, and the per-cluster copy in distributed
        shared memory (p. 213) is described there with NVIDIA's current example but not built: neither emulator models
        clusters, Metal has none, and CUDA needs compute capability 9.0. The four-pixels hint (p. 216) is mentioned, not
        implemented. A full private histogram per thread for very few bins (p. 217) is one question in{' '}
        <a href={href('ch09', '9.6')}>9.6</a>. Fig. 9.10 appears as lab 9.2's kernel at coarsening factor 1, and Fig.
        9.12 as that lab's provided comparator; the book's compile-time <code>COARSE_FACTOR</code> is a run-time argument
        in the labs. Not in the book and not done here either: the sign-aware repetition penalty, which section{' '}
        <a href={href('x1', 'X1.3')}>X1.3</a> adds.
      </p>

      <h2>Where you will meet it in real code</h2>
      <p>
        Section 9.4 showed PyTorch's <code>torch.histc</code> making the book's shared-or-global choice. Three more
        places worth reading, each this chapter's shape in production code:
      </p>
      <Beyond>
        <p>
          None of these is in the book. Read on 2026-11-20 (notes/today/ch09-facts.md D3, D4); they are snapshots, so
          follow the files, not line numbers.
        </p>
      </Beyond>
      <ul data-list="real-code">
        <li>
          CUB's <code>DeviceHistogram</code> keeps up to 256 bins in shared memory; above that it keeps one private copy
          per block in global memory, sized blocks × bins: the book's <code>bins_pool</code> of Fig. 9.9.
        </li>
        <li>
          llama.cpp's Metal top-k (<code>kernel_top_k_f32_i32</code> in <code>argsort.metal</code>) fills a 256-bin
          histogram in threadgroup memory, one 8-bit digit of each logit's bit pattern at a time, to find the bucket that
          holds the k-th value.
        </li>
        <li>
          llama.cpp's <code>count_equal</code> is thread-level privatisation pushed to one bin: each thread counts matches
          in a register over its strided elements, <code>warp_reduce_sum</code> adds up a warp, and one lane issues a
          single <code>atomicAdd</code> per warp. The Metal version sums with <code>simd_sum</code>, combines the threadgroup's
          partial sums and issues one <code>atomic_fetch_add_explicit</code>.
        </li>
      </ul>

      <h2>The project so far</h2>
      <p>
        Lab 9.3 is the counting piece of milestone M5, sampling on the GPU: per-request token counts over{' '}
        <code>[batch × vocab]</code>, updated one decode step at a time, and the frequency and presence penalties applied
        to the logits.{' '}
        <span data-m5={m5.join(',') || 'none'}>
          {backends.length === 0
            ? 'No lab results are recorded on this machine yet.'
            : m5.length
              ? `Your own lab 9.3 passes on ${m5.map(where).join(' and ')} on this machine.`
              : 'Your own lab 9.3 does not pass on any backend of this machine yet.'}
        </span>{' '}
        See the <a href={href('ch09', 'labs')}>labs</a>. Chapters 11, 12 and 14 add the pieces top-p and top-k need, and
        module X1 assembles M5 in its <a href={href('x1', 'labs')}>lab X1.3</a>.
      </p>
      <p>
        Chapter 10 keeps this chapter's shape and changes what is merged. A reduction combines everything into one value:
        the softmax's maximum and its normaliser, and the argmax of greedy decoding. Its multi-block version lets each
        block finish privately and commit once with the same <code>atomic_ref</code> and <code>fetch_add</code>, and the
        book opens it by calling reduction, like histogram, a pattern that summarises a large data set (p. 221). Start at{' '}
        <a href={href('ch10')}>chapter 10</a>.
      </p>
    </>
  );
}
