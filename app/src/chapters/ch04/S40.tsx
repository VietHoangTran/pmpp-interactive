import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source } from '../../components/notes';
import { href } from '../../lib/router';

// Prerequisite questions drawn from the notes of chapters 1 to 3.
export const questionsS40: Record<string, Question> = {
  'ch04/start/q-surplus': {
    kind: 'numeric',
    id: 'ch04/start/q-surplus',
    prompt: (
      <>
        A vector of 1000 elements is added with <code>vecAddKernel</code>, launched in blocks of 256 threads. How many
        of the launched threads have no element to work on?
      </>
    ),
    answer: 24,
    unit: 'threads',
    explain: (
      <p>
        The launch needs <code>ceil(1000 / 256) = 4</code> blocks, which is 1024 threads for 1000 elements. The last 24
        fail the test <code>if (i &lt; n)</code> and do nothing. Section 4.5 asks what those 24 cost the threads that
        share hardware with them.
      </p>
    ),
    source: 'ch. 3 §3.1, p. 46 and §3.2, p. 50',
  },
  'ch04/start/q-region4': {
    kind: 'numeric',
    id: 'ch04/start/q-region4',
    prompt: (
      <>
        Chapter 3 covered a picture 62 pixels high and 76 wide with 16 × 16 blocks, four rows of five. How many threads
        of the bottom-right block pass the range test <code>col &lt; width &amp;&amp; row &lt; height</code>?
      </>
    ),
    answer: 168,
    unit: 'threads',
    explain: (
      <p>
        The last column of blocks starts at pixel column 64, so 76 − 64 = 12 of its 16 thread columns hold a pixel. The
        last row of blocks starts at pixel row 48, so 62 − 48 = 14 of its 16 thread rows do. The corner block has both
        limits: 12 × 14 = 168 of 256. You will count this block again in section 4.5, this time warp by warp.
      </p>
    ),
    source: 'ch. 3 §3.2, pp. 54–55 (Fig. 3.5)',
  },
  'ch04/start/q-flat': {
    kind: 'numeric',
    id: 'ch04/start/q-flat',
    prompt: (
      <>
        Same picture, 76 pixels wide, 16 × 16 blocks. The thread with <code>threadIdx = (x 3, y 2)</code> in the block
        with <code>blockIdx = (x 1, y 0)</code> writes one element of <code>Pout</code>. What is its flat index?
      </>
    ),
    answer: 171,
    explain: (
      <p>
        <code>row = blockIdx.y * blockDim.y + threadIdx.y = 0 * 16 + 2 = 2</code> and{' '}
        <code>col = blockIdx.x * blockDim.x + threadIdx.x = 1 * 16 + 3 = 19</code>. Row-major: <code>row * width + col</code>{' '}
        = 2 × 76 + 19 = 171. Section 4.4 uses the same flattening, x fastest, to decide which threads of a block share a
        warp.
      </p>
    ),
    source: 'ch. 3 §3.2, p. 53',
  },
  'ch04/start/q-order': {
    kind: 'choice',
    id: 'ch04/start/q-order',
    prompt: <>A grid has two blocks, 0 and 1. What may the kernel assume about which of them runs first?</>,
    options: [
      { label: 'Block 0 runs first: blocks start in index order.', why: 'Nothing in the model says so, and section 4.2 shows why it cannot.' },
      { label: 'They always run at the same time.', why: 'They may. They may also run one after the other, in either order.' },
      { label: 'Nothing. The runtime may run them in either order, or together.', correct: true },
      { label: 'The order is fixed per device, so it can be measured once and relied on.', why: 'A test on one device shows what happened that time, not what is promised.' },
    ],
    explain: (
      <p>
        Chapter 2 stated it as a rule for the programmer: blocks work on different parts of the data and may execute in
        any order. This chapter gives the reason the rule exists, and what you get in exchange for obeying it.
      </p>
    ),
    source: 'ch. 2 §2.6, p. 39',
  },
  'ch04/start/q-throughput': {
    kind: 'choice',
    id: 'ch04/start/q-throughput',
    prompt: (
      <>
        A CPU spends much of its area on large caches and branch predictors. A GPU spends it on many simple cores
        instead. Why is that a sound trade for a GPU?
      </>
    ),
    options: [
      {
        label: 'A GPU is built for instruction throughput across many threads, and caches and predictors mostly buy latency.',
        correct: true,
      },
      { label: 'GPU programs rarely branch and rarely reuse data, so caches and predictors would sit idle.', why: 'Kernels branch (every range test is one) and reuse data constantly.' },
      { label: 'GPU memory is fast enough that a cache would add nothing a kernel could measure.', why: 'The opposite: an access to global memory is the slow operation of this chapter.' },
      { label: 'Simple cores run each thread faster than a complex core would, so the area is better spent.', why: 'Each thread runs slower. There are many more of them.' },
    ],
    explain: (
      <p>
        Chapter 1's argument: halving latency costs far more area and power than doubling throughput by adding units.
        The trade only works if the hardware always has another thread to run while one waits. Section 4.6 shows the
        mechanism: an SM keeps many more threads than it can execute and switches among them at no cost.
      </p>
    ),
    source: 'ch. 1 §1.1, pp. 4–5, recalled on p. 67',
  },
};

export function S40() {
  const q = questionsS40;
  return (
    <>
      <p>
        Chapters 2 and 3 taught you to launch a grid: so many blocks of so many threads, each thread finding its own
        piece of the data. They said almost nothing about what the device does with that grid once it has it. This
        chapter does. After it, the two numbers in <code>&lt;&lt;&lt;grid, block&gt;&gt;&gt;</code> are decisions you
        can reason about, not values copied from an example.
      </p>
      <Source pages="67" />
      <p>The chapter follows a grid downwards, in the order of the book's pages:</p>
      <ul>
        <li>
          <b>blocks onto hardware</b>: what the device is made of (<a href={href('ch04', '4.1')}>4.1</a>) and how the
          runtime hands it whole blocks (<a href={href('ch04', '4.2')}>4.2</a>);
        </li>
        <li>
          <b>barriers</b>, and why blocks may run in any order (<a href={href('ch04', '4.3')}>4.3</a>);
        </li>
        <li>
          <b>blocks cut into warps</b>, the groups of threads that execute together (
          <a href={href('ch04', '4.4')}>4.4</a>);
        </li>
        <li>
          <b>warps at a branch</b>: what it costs when threads of one warp disagree (
          <a href={href('ch04', '4.5')}>4.5</a>);
        </li>
        <li>
          <b>warps waiting on memory</b>, and why a device keeps more of them than it can run (
          <a href={href('ch04', '4.6')}>4.6</a>);
        </li>
        <li>
          <b>how many fit</b>: the budgets that decide it (<a href={href('ch04', '4.7')}>4.7</a>);
        </li>
        <li>
          <b>asking the device</b> for its sizes before you choose a launch (<a href={href('ch04', '4.8')}>4.8</a>).
        </li>
      </ul>
      <p>
        Memory comes next: chapter 5 is the memory on the chip, chapter 6 the memory off it and the tuning that follows
        from both.
      </p>

      <h2>Where this sits under a framework call</h2>
      <p>
        When <code>torch.matmul</code> runs on a GPU tensor, some kernel does the work, and whoever wrote that kernel
        chose a block size and a grid for it. Those choices are made in this chapter's vocabulary, and you can see it in
        the sources this course is heading for. PyTorch's CUDA kernels use a constant <code>C10_WARP_SIZE</code>.
        llama.cpp's CUDA backend has <code>#define WARP_SIZE 32</code>. MLX's Metal softmax has{' '}
        <code>constexpr int SIMD_SIZE = 32</code>. Each of those lines is a statement about how a device groups threads
        for execution. By section 4.4 you will be able to say what each one assumes, and by section 4.8 what it should
        have asked the device instead.
      </p>
      <Beyond>
        <p>
          The book does not mention PyTorch, llama.cpp or MLX. The three constants were read in the projects' sources on
          2026-11-20 (notes/today/ch04-facts.md, D1, D3, D4).
        </p>
      </Beyond>

      <h2>Three words, three vendors</h2>
      <p>
        The book teaches CUDA, and its words are NVIDIA's. The same three ideas carry this whole chapter on every
        backend, under different names:
      </p>
      <Beyond>
        <p>
          Setting the columns side by side is ours. No vendor draws these equivalences, and the middle row is loose:
          section 4.1 says how. Names checked against vendor documentation (notes/today/ch04-facts.md, B1, B2, C2).
        </p>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }} aria-label="Vocabulary across vendors">
        <thead>
          <tr>
            <th>What it is</th>
            <th>CUDA (the book)</th>
            <th>HIP on AMD</th>
            <th>Metal</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>The group of threads you launch together, which can share memory and meet at a barrier</td>
            <td>block</td>
            <td>block in HIP code; workgroup in AMD's hardware documentation</td>
            <td>threadgroup</td>
          </tr>
          <tr>
            <td>The piece of hardware one such group runs on</td>
            <td>streaming multiprocessor (SM)</td>
            <td>compute unit (CU)</td>
            <td>not documented</td>
          </tr>
          <tr>
            <td>The threads inside a group that execute one instruction together</td>
            <td>warp</td>
            <td>warp in HIP's documentation; wavefront in AMD's instruction-set terms</td>
            <td>SIMD-group</td>
          </tr>
        </tbody>
      </table>

      <h2>What you will measure</h2>
      <p>
        Four <a href={href('ch04', 'labs')}>labs</a> go with the chapter: ask the device for its sizes and derive a
        launch from them; measure what a divergent warp costs; hunt three defects around a barrier; and sweep the block size of a
        kernel that spends its time waiting on memory.
      </p>
      <p>
        One limit to know before you start. The CPU emulators run your kernel's logic: they catch wrong results, a
        barrier in the wrong place, a race. They do not run the threads of a warp in step and they do not wait on memory, so they cannot
        show what divergence or low occupancy costs. Those two measurements need a GPU, and the numbers you get belong
        to that GPU.
      </p>
      <Aside>
        <p>
          <code>python3 labs/pmpp.py doctor</code> lists the backends this machine can run. Anything a lesson says about
          "your device" comes from that report.
        </p>
      </Aside>

      <h2>Before you start</h2>
      <p>
        Five questions on what this chapter assumes from chapters 1 to 3. Each one is used again later in the chapter,
        and each comes back in <a href={href('review')}>review</a>.
      </p>
      <Ask q={q['ch04/start/q-surplus']} label="From chapter 3" />
      <Ask q={q['ch04/start/q-region4']} label="From chapter 3" />
      <Ask q={q['ch04/start/q-flat']} label="From chapter 3" />
      <Ask q={q['ch04/start/q-order']} label="From chapter 2" />
      <Ask q={q['ch04/start/q-throughput']} label="From chapter 1" />
    </>
  );
}
