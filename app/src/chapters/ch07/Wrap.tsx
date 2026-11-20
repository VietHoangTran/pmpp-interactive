import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Beyond, Source, type Tier } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';

// A kernel of our own, for the first question. Checked copy: labs/ch07/snippets/wrap_causal_conv1d.cu
// (Wrap.test.ts ties the two, runs a model of the kernel and the file itself on the emulator for
// every n up to 400, and pins the numbers the other questions state).
const CAUSAL_CONV1D = `// A causal 1D convolution: y[i] = w[0]*x[i-3] + w[1]*x[i-2] + w[2]*x[i-1] + w[3]*x[i],
// with x[j] = 0 for j < 0. Each output looks back K - 1 inputs, so the halo is one-sided.
// One thread per output; grid = ceil(n / BLOCK) blocks of BLOCK threads.
#define BLOCK 128
#define K 4

__constant__ float w[K];

__global__ void causalConv1d(const float* x, float* y, int n) {
    __shared__ float tile[BLOCK + K - 1];
    int t = threadIdx.x;
    int i = blockIdx.x * BLOCK + t;
    if (i >= n) return;                     // no output past the end

    tile[t + K - 1] = x[i];                 // this thread's own input
    if (t < K - 1) {                        // the halo: the K - 1 inputs before the block
        int j = i - (K - 1);
        tile[t] = (j >= 0) ? x[j] : 0.0f;   // before the start of x: a ghost zero
    }
    __syncthreads();

    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += w[k] * tile[t + k];
    y[i] = acc;
}`;
const CAUSAL_TIERS: Record<string, Tier> = { x: 'global', y: 'global', tile: 'shared', w: 'const', t: 'reg', i: 'reg', j: 'reg', k: 'reg', acc: 'reg' };

// ---------------------------------------------------------------- the numbers the questions state
/** The causal kernel's sizes, and the remainders n mod BLOCK for which it is wrong (pinned by Wrap.test.ts). */
export const CAUSAL = { block: 128, taps: 4, wrongRemainders: [1, 2] } as const;

/** Exercise 7 on one case: a 1000 x 1000 image, a 5 x 5 filter, 28 x 28 output tiles. */
export const EX7 = { n: 1000, m: 5, t: 28 } as const;
export const EX7_BLOCKS = Math.ceil(EX7.n / EX7.t) ** 2;
export const EX7_THREADS_712 = (EX7.t + EX7.m - 1) ** 2; // Fig. 7.12: the block is the input tile
export const EX7_THREADS_715 = EX7.t ** 2; // Fig. 7.15: the block is the tile
export const EX7_EXTRA = EX7_BLOCKS * (EX7_THREADS_712 - EX7_THREADS_715);

/** Section 7.5's count for one interior block, in 3D: t^3 outputs, an (t + m - 1)^3 input tile, 4-byte floats. */
export const ai3d = (t: number, m: number) => (2 * t ** 3 * m ** 3) / (4 * ((t + m - 1) ** 3 + t ** 3));
/** Exercise 10 with a 5 x 5 x 5 filter: the largest cube of threads within 1024 is 10 on a side. */
export const EX10 = { maxThreads: 1024, m: 5, inTile: Math.floor(Math.cbrt(1024 + 0.5)) } as const;
export const EX10_OUT = EX10.inTile - (EX10.m - 1);
export const EX10_AI = ai3d(EX10_OUT, EX10.m);
export const EX10_IDEAL = EX10.m ** 3 / 4;

const n = (v: number) => v.toLocaleString('en-US');

export const questionsWrap: Record<string, Question> = {
  'ch07/wrap/q-early-return': {
    kind: 'choice',
    id: 'ch07/wrap/q-early-return',
    prompt: (
      <>
        The kernel above computes a short causal convolution, the shape of section 7.5&rsquo;s LLM-serving example. For
        which lengths <code>n</code> does it produce a wrong output?
      </>
    ),
    options: [
      {
        label: 'For none: the threads that return own no output, and the barrier does not wait for them.',
        why: 'The barrier indeed does not wait for them. But which threads write tile[0], tile[1] and tile[2], and do they all still exist in the last block?',
      },
      {
        label: 'Whenever n is not a multiple of 128.',
        why: 'With 3 or more outputs in the last block, threads 0, 1 and 2 all exist and load the whole halo. A missing thread past the end owns no slot that a valid output reads.',
      },
      { label: 'When the last block has only 1 or 2 outputs: n = 129 and n = 130 are wrong, n = 131 is right.', correct: true },
      {
        label: 'For every n: a barrier after an early return hangs the block.',
        why: "NVIDIA's guide says a barrier waits for the threads that have not exited, and Apple says the same for Apple silicon (ch04-facts A6, C3). The kernel finishes; the question is what it read.",
      },
    ],
    explain: (
      <p>
        Threads past the end return before loading anything, and nothing hangs. The damage is in what they did not
        load. The halo slots <code>tile[0]</code>, <code>tile[1]</code> and <code>tile[2]</code> are written by threads 0,
        1 and 2. When the last block has 3 or more outputs those threads exist, and every slot a valid output reads has
        been written. With 1 or 2 outputs, thread 2 (and perhaps thread 1) has returned, and output 0 reads a slot nobody
        wrote: on a GPU whatever was there, on the emulator <code>0xCD</code> bytes. So the kernel is wrong exactly when{' '}
        <code>n mod 128</code> is 1 or 2. The fix is section 7.5&rsquo;s structure: no early return; every thread loads
        (a ghost zero past the end), reaches the barrier, and only then tests <code>i &lt; n</code> before computing and
        storing. AMD documents no rule for exited threads at a barrier, one more reason to bring every thread to it.
      </p>
    ),
    source:
      'a kernel of ours (labs/ch07/snippets/wrap_causal_conv1d.cu, run on the emulator for n = 1 to 400); barrier rule: ch04-facts A6, C3; the causal shape: ch07-facts D3',
  },
  'ch07/wrap/q-replicate': {
    kind: 'choice',
    id: 'ch07/wrap/q-replicate',
    prompt: (
      <>
        You need lab 7.1&rsquo;s basic kernel to treat the edges like PyTorch&rsquo;s{' '}
        <code>padding_mode=&apos;replicate&apos;</code>: a missing neighbour takes the value of the nearest pixel inside
        the image. What is the smallest change to the kernel?
      </>
    ),
    options: [
      {
        label: 'Clamp inRow and inCol into the image instead of testing them, and accumulate every tap.',
        correct: true,
      },
      {
        label: 'Keep the ghost test, and for every skipped tap add the nearest edge pixel to Pvalue.',
        why: 'Each skipped tap has its own weight. The replicated value has to be multiplied by that tap’s weight, which is what reading the clamped element does.',
      },
      {
        label: 'Flip the filter for threads near an edge, so that their taps fold back into the image.',
        why: 'Flipping changes which weight meets which input; it replicates nothing.',
      },
      {
        label: 'Change nothing in the kernel: launch r extra threads past each edge to supply the copies.',
        why: 'Extra threads compute outputs that do not exist (the output guard drops them), and the real edge outputs still skip their ghost taps.',
      },
    ],
    explain: (
      <p>
        The clamp is <code>inRow = min(max(inRow, 0), height - 1)</code>, and the same for <code>inCol</code> with{' '}
        <code>width</code>. With it, every tap reads a real element, and the zero policy&rsquo;s <code>if</code> disappears. There
        are no skipped taps, so at the source level every lane of an edge warp runs the same multiply-adds; the output
        guard stays. The book names the replicate policy but gives no kernel for it; the clamp is ours. Textures can do
        the same in hardware (CUDA&rsquo;s Clamp address mode, Metal&rsquo;s <code>clamp_to_edge</code>), which section
        7.1 mentions and the course, like the book, does not use.
      </p>
    ),
    source: 'ch. 7 §7.1, p. 162 (the policies) and §7.2, p. 166 (the ghost test); the clamp is ours; ch07-facts D1, A6, C5',
  },
  'ch07/wrap/q-warp-index': {
    kind: 'choice',
    id: 'ch07/wrap/q-warp-index',
    prompt: (
      <>
        A kernel with 256-thread blocks reads a <code>__constant__</code> table at index{' '}
        <code>threadIdx.x / 32</code>. How many distinct addresses does one warp read, with 32-wide warps and with
        64-wide wavefronts?
      </>
    ),
    options: [
      { label: '1 and 1.', why: 'A 64-wide wavefront covers threadIdx.x from 0 to 63 (or 64 to 127, ...): two values of threadIdx.x / 32.' },
      { label: '1 and 2.', correct: true },
      { label: '32 and 64.', why: 'All 32 lanes of a warp that starts at a multiple of 32 share threadIdx.x / 32.' },
      { label: '8 and 8.', why: 'threadIdx.x / 32 counts warps of 32 within the block; it does not depend on the block having 8 of them.' },
    ],
    explain: (
      <p>
        On a 32-wide warp every lane computes the same index: one address, the case NVIDIA says is as fast as a register.
        A 64-wide wavefront spans two values, and AMD&rsquo;s scalar cache serves <code>__constant__</code> reads only
        when they are uniform across the wavefront; otherwise they are serialised. The literal 32 is the bug: index by{' '}
        <code>threadIdx.x / warpSize</code>, or by something uniform per block such as <code>blockIdx.x</code>, and the
        read is one address on both.
      </p>
    ),
    source: 'ch. 7 §7.4, p. 171; ch07-facts A3, B1; wavefront widths ch05-facts B5',
  },
  'ch07/wrap/q-ex7': {
    kind: 'numeric',
    id: 'ch07/wrap/q-ex7',
    prompt: (
      <>
        Exercise 7 on one case: a {n(EX7.n)} × {n(EX7.n)} image, a {EX7.m} × {EX7.m} filter and {EX7.t} × {EX7.t}{' '}
        output tiles. In all, how many more threads does Fig. 7.12 launch than Fig. 7.15 with the same output tile?
      </>
    ),
    answer: EX7_EXTRA,
    unit: 'threads',
    explain: (
      <p>
        Both kernels have one block per output tile: ⌈{n(EX7.n)} / {EX7.t}⌉ = {Math.ceil(EX7.n / EX7.t)} per side,{' '}
        {n(EX7_BLOCKS)} blocks. Fig. 7.12&rsquo;s block is the input tile, ({EX7.t} + {EX7.m} − 1)² = {n(EX7_THREADS_712)}{' '}
        threads, with as many floats of shared memory ({n(4 * EX7_THREADS_712)} bytes). Fig. 7.15&rsquo;s block is the tile,{' '}
        {n(EX7_THREADS_715)} threads and {n(4 * EX7_THREADS_715)} bytes. The difference is {n(EX7_BLOCKS)} ×{' '}
        {EX7_THREADS_712 - EX7_THREADS_715} = {n(EX7_EXTRA)} threads, every one of which loads a halo element and then
        idles.
      </p>
    ),
    source: 'ch. 7 Exercise 7, p. 181; Figs. 7.12 and 7.15, pp. 174, 178',
  },
  'ch07/wrap/q-3d': {
    kind: 'numeric',
    id: 'ch07/wrap/q-3d',
    prompt: (
      <>
        Exercise 10 asks for Fig. 7.12 in 3D. Take a {EX10.m} × {EX10.m} × {EX10.m} filter, at most {n(EX10.maxThreads)}{' '}
        threads per block, and the largest cube of threads that fits as the input tile. What does section 7.5&rsquo;s
        count give for an interior block, in FLOP/B?
      </>
    ),
    answer: Math.round(EX10_AI * 100) / 100,
    tol: 0.015,
    unit: 'FLOP/B',
    explain: (
      <p>
        {EX10.inTile}³ = {n(EX10.inTile ** 3)} threads fit and {EX10.inTile + 1}³ = {n((EX10.inTile + 1) ** 3)} do not,
        so the input tile is {EX10.inTile} on a side and the output tile {EX10_OUT}. Per block: 2 · {EX10_OUT}³ · {EX10.m}³
        = {n(2 * EX10_OUT ** 3 * EX10.m ** 3)} FLOP, against {n(EX10.inTile ** 3)} floats loaded and {n(EX10_OUT ** 3)}{' '}
        stored, {n(4 * (EX10.inTile ** 3 + EX10_OUT ** 3))} bytes: {EX10_AI.toFixed(1)} FLOP/B, about a third of the 3D
        ideal m³/4 = {EX10_IDEAL}. In 2D the same filter and block budget reach 5.42 of 6.25. A halo is a shell, and
        around a cube this small most of the tile is shell: {n(EX10.inTile ** 3 - EX10_OUT ** 3)} of{' '}
        {n(EX10.inTile ** 3)} elements. Chapter 8 is about getting past this limit.
      </p>
    ),
    source: 'ch. 7 Exercise 10, p. 181; the count of §7.5, p. 176, extended to 3D (ours); Fig. 7.14, p. 177',
  },
  'ch07/wrap/q-same-speed': {
    kind: 'choice',
    id: 'ch07/wrap/q-same-speed',
    prompt: (
      <>
        On some GPU, lab 7.1&rsquo;s constant-memory kernel runs within a few percent of the basic kernel at every
        radius. Which reading fits best?
      </>
    ),
    options: [
      {
        label: 'The compiler ignored the __constant__ qualifier, so the filter still came from DRAM on every tap.',
        why: 'The qualifier is not a hint: kernels cannot write F_c, and NVIDIA serves it through the constant cache. The question is how much DRAM traffic there was to remove.',
      },
      {
        label: 'Ordinary caches already held the filter for the basic kernel, so little DRAM traffic was left to remove.',
        correct: true,
      },
      {
        label: 'Both kernels are compute-bound, since 0.25 and 0.5 FLOP/B are above this device\'s threshold.',
        why: 'That needs a threshold below 0.25 FLOP/B: a device that moves more than four bytes for every operation it can do. Lab 5.1 measures yours.',
      },
      {
        label: 'The constant cache only pays off for filters larger than its 8 KB working set, and this one is small.',
        why: 'The opposite: a filter that fits in the working set is the best case.',
      },
    ],
    explain: (
      <p>
        The model&rsquo;s factor of 2 assumes every read of <code>F</code> reaches DRAM. But every thread reads the same
        few hundred bytes, which any cache keeps after the first touch, as the book itself says (p. 168). Constant memory
        then changes which cache answers, not how much reaches DRAM. What remains is the input, one load of{' '}
        <code>N</code> per tap, and that is what tiling goes after.
      </p>
    ),
    source: 'ch. 7 §7.3, p. 168 and §7.4, p. 171; ch07-facts A3; lab 7.1',
  },
};

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = realBackends(reports).filter((b) => b !== 'race');
  const passing = (lab: string) => backends.filter((b) => labStatus(reports[b], lab, 'work') === 'pass');
  const l1 = passing('ch07/lab1_constconv');
  const l2 = passing('ch07/lab2_tiledconv');
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch07/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const labLine = (on: string[]) => (on.length ? `passes on ${on.join(' and ')}` : 'has not passed on a device here yet');
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul>
        <li>
          A convolution output is a weighted sum over a patch. The book&rsquo;s formula does not flip the filter, which
          is what PyTorch computes too. What a missing neighbour is worth (zero, the nearest edge value, a wrap) is a
          policy you choose.
        </li>
        <li>
          One thread per output is correct and simple: a guard for the output, and a test (or a clamp) for every tap.
          Which warps diverge at the edges depends on the block shape and the warp width; read the width, never write
          32.
        </li>
        <li>
          The ideal is m²/4 FLOP/B in 2D and m/4 in 1D for 4-byte values, and it scales up as the element shrinks. The
          basic kernel asks for 0.25 whatever the filter: the gap is the reuse it throws away. A small filter cannot be
          compute-bound, however good the kernel.
        </li>
        <li>
          Constant memory, or an argument passed by value, suits small read-only data that every lane of a warp reads at
          the same index. Per-lane indices are its worst case. Keep a <code>__constant__</code> symbol, its kernel and
          its copy in one file.
        </li>
        <li>
          When the input tile is bigger than the output tile, the halo costs: the intensity is the ideal times a factor
          that approaches 1 only for tiles several times wider than the filter, and the block size caps the tile. Store
          the ghost zeros, use one barrier that every thread reaches, and convert unsigned indices to <code>int</code>{' '}
          before subtracting.
        </li>
        <li>
          Leaving the halo to the cache gives tiles that are a power of two and keeps every lane busy, but its traffic
          depends on a cache you do not control, and it splits warps inside the loop.
        </li>
        <li>
          The models count what a kernel asks of memory. Caches decide what reaches DRAM, and only a measurement says
          how much: predict, then run the labs.
        </li>
      </ul>
      <Source section="7.7" pages="179–180" />
      <p>
        The book closes by pointing ahead. Stencils, the subject of chapter 8, are convolutions with a special shape,
        and chapter 8 also takes on the limit that the block size puts on the tile. Convolutional layers, and the way
        libraries turn a convolution into a matrix multiplication, are chapter 19.
      </p>
      <Aside>
        <p>
          Left out on purpose: the general tour of CPU caches on pp. 170 to 171 is reduced to what NVIDIA, AMD and Apple
          document today (section 7.4&rsquo;s table), with no latency animation, because no L2 latency figure could be
          checked. Fig. 7.8&rsquo;s review of the memory model (p. 168) is chapter 5&rsquo;s section 5.2.
        </p>
      </Aside>

      <h2>A mixed check</h2>
      <p>
        These draw on the whole chapter and on situations the sections did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly on the first recorded try.` : ''}
      </p>
      <Code
        tabs={[
          {
            label: 'CUDA (not a lab file: for this question only)',
            code: CAUSAL_CONV1D,
            tiers: CAUSAL_TIERS,
            status: 'front-end-checked; emulator-checked',
          },
        ]}
      />
      <Ask q={questionsWrap['ch07/wrap/q-early-return']} label="Find the defect" />
      <Ask q={questionsWrap['ch07/wrap/q-replicate']} />
      <Ask q={questionsWrap['ch07/wrap/q-warp-index']} />
      <Ask q={questionsWrap['ch07/wrap/q-ex7']} label="Exercise 7" />
      <Ask q={questionsWrap['ch07/wrap/q-3d']} label="Exercise 10" />
      <Ask q={questionsWrap['ch07/wrap/q-same-speed']} />

      <h2>The book&rsquo;s exercises</h2>
      <p>
        The chapter ends with eleven exercises (pp. 180–181). Each is worked or set up somewhere in this course; the
        book&rsquo;s own wording is worth doing as well.
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
              <td>y[0] of Fig. 7.3</td>
              <td>
                the prediction before the sliding window in <a href={href('ch07', '7.1')}>7.1</a>, which hides y[0] until
                you commit
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td>A 1D convolution by hand</td>
              <td>
                a question and an explorer preset in <a href={href('ch07', '7.1')}>7.1</a>
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td>What five 1D filters do</td>
              <td>
                the explorer&rsquo;s presets and a question in <a href={href('ch07', '7.1')}>7.1</a>
              </td>
            </tr>
            <tr>
              <td>4, 5, 6</td>
              <td>Ghost cells and multiplications in 1D, N×N and N1×N2</td>
              <td>
                the ghost counter and its formulas in <a href={href('ch07', '7.1')}>7.1</a>, with a question on exercise 5
              </td>
            </tr>
            <tr>
              <td>7</td>
              <td>Blocks, threads and shared memory for Figs. 7.12 and 7.15</td>
              <td>
                the Exercise 7 question above; tile sizes in the explorer of <a href={href('ch07', '7.5')}>7.5</a>
              </td>
            </tr>
            <tr>
              <td>8, 9</td>
              <td>Figs. 7.7 and 7.9 in 3D</td>
              <td>not coded; the changes are listed below, and lab 7.1 is the place to port them</td>
            </tr>
            <tr>
              <td>10</td>
              <td>Fig. 7.12 in 3D</td>
              <td>sizing and intensity only: the Exercise 10 question above</td>
            </tr>
            <tr>
              <td>11</td>
              <td>The block as the output tile</td>
              <td>
                the explorer&rsquo;s second organisation and two questions in <a href={href('ch07', '7.5')}>7.5</a>; MLX&rsquo;s
                depthwise convolution is built this way
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        For exercises 8 and 9: add a <code>depth</code> argument and a third index from <code>blockIdx.z</code>,{' '}
        <code>blockDim.z</code> and <code>threadIdx.z</code>; add a third filter loop and a bounds test in z; read{' '}
        <code>N[(inZ*height + inRow)*width + inCol]</code>; and flatten the filter over (2r + 1)³ weights. In lab 7.1 the
        largest radius, 7, would then need 15³ = 3,375 floats, 13,500 bytes: inside CUDA&rsquo;s 64 KB of constant memory
        but past the 8 KB working set of its cache, and past Metal&rsquo;s 4 KB for <code>setBytes</code>. The emulator
        runs 3D blocks, so the port can be tested without a GPU.
      </p>
      <Beyond>
        <p>The 3D sizes are our arithmetic, set against ch07-facts A3 and C1.</p>
      </Beyond>

      <h2>Where the PDF looks wrong</h2>
      <p>
        The copy this course was built from has these apparent slips in chapter 7, listed so that you are not thrown by
        them when you read the book alongside. Where one affects code, the labs use corrected, tested code and say where
        it differs from the page.
      </p>
      <ul data-list="errata">
        <li>
          p. 159 says Gaussian filters can sharpen edges. In the tools you use a Gaussian filter is a blur (torchvision
          documents <code>GaussianBlur</code> as one), and this course calls it smoothing.
        </li>
        <li>p. 161 calls the inner product for y[3] the calculation &ldquo;for x[3]&rdquo;.</li>
        <li>
          p. 163 names the filter M in the term-by-term expansion of P[2][2], where the figure calls it f, and prints the
          third row&rsquo;s second term as N<sub>2,1</sub>·M<sub>1,1</sub> where the pattern needs M<sub>2,1</sub>. The
          numbers and the sum, 321, are right.
        </li>
        <li>p. 165 says the kernel receives five arguments, then lists six.</li>
        <li>
          Fig. 7.7 does not compile: it indexes the plain pointers F and P with two subscripts, and inRow and inCol have
          no type. It also has no output guard. <a href={href('ch07', '7.2')}>Section 7.2</a> hunts these; lab 7.1 has
          the corrected kernel.
        </li>
        <li>
          Fig. 7.9 fixes F&rsquo;s size with a compile-time radius but still takes the radius r at run time; the two
          must agree. Lab 7.1 sizes F_c for radius 7 and indexes it flat.
        </li>
        <li>
          p. 168 says most filters have a radius of 7 or less, then bounds a 3D filter by 7³ = 343 elements, which is a
          filter 7 wide, not radius 7 (15³).
        </li>
        <li>
          p. 169 suggests an <code>extern</code> declaration when the kernel and the host code are in different files.
          In nvcc&rsquo;s default whole-program mode that is an error; it needs <code>-rdc=true</code>. The same page
          says <code>cudaMemcpyToSymbol</code> tells the runtime the data will not change; NVIDIA documents it as a copy
          to any device symbol, and read-only comes from <code>__constant__</code>.
        </li>
        <li>
          p. 171 puts L1 at &ldquo;typically up to 128 KB&rdquo;: NVIDIA&rsquo;s unified L1 and shared memory is now 96 to
          336 KB per SM. It also has &ldquo;high-thought writes&rdquo; for high-throughput.
        </li>
        <li>p. 172 has a stray &ldquo;or&rdquo; in &ldquo;the input tiles are or 4× larger&rdquo;.</li>
        <li>
          Figs. 7.12 and 7.15 declare the filter as F_c and use it as F, declare N_s without an element type, and write
          0.0 for 0.0f. Lab 7.2 fixes all of these.
        </li>
        <li>
          p. 175 says thread (5,5) uses the patch whose corner is N_s[5][5]; with radius 1 that is thread (6,6), the
          one Fig. 7.13 draws (<a href={href('ch07', '7.5')}>section 7.5</a> asks). The same paragraph has
          &ldquo;threadId.y&rdquo;.
        </li>
        <li>
          p. 176 introduces the intensity as operations per byte the block loads, but the denominator rightly counts
          its stores too.
        </li>
        <li>
          Fig. 7.15, line 21, reads N_s without the <code>-FILTER_RADIUS</code> its own guard uses, so every shared tap
          reads the wrong element and some fall past the tile. Its <code>&gt;= 0</code> tests are on unsigned values
          and always true; the guard still works because the other half rejects the wrapped values. Its line numbers
          are corrupted (24 to 27 appear twice). <a href={href('ch07', '7.6')}>Section 7.6</a> and lab 7.2&rsquo;s part B
          cover all three.
        </li>
        <li>p. 178 is missing a word in &ldquo;by the time a block needs to its halo cells&rdquo;.</li>
        <li>
          p. 179 sends you to chapter 20 for convolutional neural networks; footnote 1 and the contents put them in
          chapter 19.
        </li>
        <li>Exercise 1 asks for P[0] in Fig. 7.3, whose output array is called y.</li>
      </ul>
      <Aside>
        <p>
          These are our readings of one PDF. If your copy differs, trust your copy. The items about extern, the copy
          call and the L1 size come from checking the book against current documentation (ch07-facts A1, A2, A5, D5).
        </p>
      </Aside>

      <h2>The project so far</h2>
      <p data-project={[l1.length ? 'lab1' : '', l2.length ? 'lab2' : ''].filter(Boolean).join(',') || 'none'}>
        Chapter 7 adds no milestone to the through-line project, which stands at M2: the tiled, boundary-checked GEMM of
        chapters 5 and 6. What the project takes from this chapter is technique: arguments by value for every small
        parameter block, the intensity count for sizing tiles, and halo tiling for any sliding window.{' '}
        {backends.length === 0
          ? 'No lab results are recorded on this machine yet.'
          : `On this machine your lab 7.1 ${labLine(l1)}, and your lab 7.2 ${labLine(l2)}.`}{' '}
        See the <a href={href('ch07', 'labs')}>labs</a>.
      </p>
      <p>
        <a href={href('ch08', 'start')}>Chapter 8</a> picks up where this one stops. A stencil is a convolution whose
        weights come from a differential equation and whose points lie along the axes, usually in 3D, where the
        block-size limit you met in exercise 10 bites hardest. Its answers are thread coarsening, which lets one block
        march through a tile deeper than its threads, and register tiling.
      </p>
    </>
  );
}
