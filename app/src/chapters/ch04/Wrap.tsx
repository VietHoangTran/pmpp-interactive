import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Beyond, Source } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { capacity, waves } from '../../sim/blocksched';
import { PRESETS, warpsNeeded } from '../../sim/latency';
import { BOOK, occupancy, pct, type Limits } from '../../sim/occupancy';
import { classifyBlock, dims } from '../../sim/warps';

// A kernel of our own, for one question. Checked copy: labs/ch04/snippets/warp_tail_sum.cu
// (Wrap.snippets.test.ts ties the two and pins the numbers the other questions take from the models).
const WARP_TAIL_SUM = `// Sum of each block's 256 values. The last six steps are written for one warp, with no barrier.
#define BLOCK 256

__global__ void blockSum(const float* in, float* out) {
    __shared__ float s[BLOCK];
    unsigned int t = threadIdx.x;
    s[t] = in[blockIdx.x * BLOCK + t];
    __syncthreads();
    for (unsigned int stride = BLOCK / 2; stride > 32; stride /= 2) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    if (t < 32) {
        volatile float* v = s;
        v[t] += v[t + 32];
        v[t] += v[t + 16];
        v[t] += v[t + 8];
        v[t] += v[t + 4];
        v[t] += v[t + 2];
        v[t] += v[t + 1];
    }
    if (t == 0) out[blockIdx.x] = s[0];
}`;

// ---------------------------------------------------------------- the numbers the questions take from the models
/** A 16 x 8 tile that skips its rows from K on: divergent warps at widths 32 and 64. */
export const TILE = dims(16, 8);
export const ROW_LIMITS = [2, 4, 6] as const;
export const divergentWarps = (k: number, width: number) =>
  classifyBlock(TILE, width, (_x, y) => y < k).states.filter((s) => s === 'divergent').length;

/** Section 4.6's load-heavy preset, then the same loop with four times the arithmetic per load. */
export const BEFORE_PACK = PRESETS.loadHeavy;
export const AFTER_PACK = { work: 4 * PRESETS.loadHeavy.work, latency: PRESETS.loadHeavy.latency };
export const NEEDED_BEFORE = warpsNeeded(BEFORE_PACK.work, BEFORE_PACK.latency);
export const NEEDED_AFTER = warpsNeeded(AFTER_PACK.work, AFTER_PACK.latency);

/** A small grid on a device described only by its limits (no product). */
export const SMALL_GRID = { blocks: 64, threadsPerBlock: 256, units: 40, slotsBefore: 2, slotsAfter: 4 } as const;
export const CAP_BEFORE = capacity(SMALL_GRID.units, SMALL_GRID.slotsBefore);
export const CAP_AFTER = capacity(SMALL_GRID.units, SMALL_GRID.slotsAfter);
export const WARPS_IN_GRID = SMALL_GRID.blocks * (SMALL_GRID.threadsPerBlock / BOOK.width!);

/** One kernel, two sets of limits: the book's example and an SM with 1536 thread slots and 16 block slots. */
export const KERNEL_40 = { threads: 256, regs: 40, shared: 0 };
export const SMALLER_SM: Limits = { ...BOOK, threadSlots: 1536, blockSlots: 16, shared: null };
export const ON_BOOK = occupancy({ ...BOOK, shared: null }, KERNEL_40);
export const ON_SMALLER = occupancy(SMALLER_SM, KERNEL_40);

const n = (v: number) => v.toLocaleString('en-US');

export const questionsWrap: Record<string, Question> = {
  'ch04/wrap/q-warpsync': {
    kind: 'choice',
    id: 'ch04/wrap/q-warpsync',
    prompt: (
      <>
        The kernel above leaves out the barrier for its last six steps, because "32 threads are one warp, and a warp
        runs in step". What does this chapter say about those six lines?
      </>
    ),
    options: [
      {
        label: 'They are correct: threads 0 to 31 form one warp, and a warp executes one instruction for all its threads at once.',
        why: "That is section 4.4's SIMD picture (p. 76). Section 4.5 adds that from Volta on you cannot assume the threads of a warp stay together (p. 81), and NVIDIA's guide calls warp-synchronous assumptions invalid.",
      },
      {
        label: 'They fail on a device with 64-wide warps, where threads 0 to 31 are only half a warp.',
        why: 'Half a warp is still inside one warp, and AMD, whose gfx9 devices are 64 wide, documents that a warp runs in lockstep. The width this code cannot survive is one narrower than 32, where threads 0 to 31 would be two groups.',
      },
      {
        label: 'They assume threads 0 to 31 run in step as one group, which only AMD promises; each step needs synchronising.',
        correct: true,
      },
      {
        label: 'They are correct because the barrier at the end of the loop has already synchronised the block.',
        why: 'That barrier orders the loop. In the six lines after it, each thread reads a value that another thread of the same group writes in the same line, with nothing between them.',
      },
    ],
    explain: (
      <>
        <p>
          In each line thread <code>t</code> reads <code>v[t + d]</code>, which thread <code>t + d</code> writes in the
          same line. That is correct only if every thread reads before any thread writes, which is what lockstep would
          give. Section 4.4 says to assume the threads of a block run in any order and to put a barrier between phases
          (p. 74); section 4.5 says that from Volta on the threads of one warp may not stay together and points to{' '}
          <code>__syncwarp()</code> (p. 81). Exercise 5 asked the same question about <code>__syncthreads()</code>.
        </p>
        <p>
          Only AMD promises lockstep; NVIDIA has not since compute capability 7.0, and Apple documents neither lockstep
          nor a width. On CUDA the repair is explicit warp synchronisation between the read and the write of each step;
          written for every backend, it is a barrier the whole block reaches, or the warp-level functions chapters 10 to
          12 use.
          The width 32 is a second assumption, and the same source runs on 64-wide warps under HIP.
        </p>
      </>
    ),
    source:
      'ch. 4 §4.4 p. 74, §4.5 p. 81, exercise 5 p. 91; notes/today/ch04-facts.md A10, B5, C2; the kernel is ours',
  },
  'ch04/wrap/q-warpbarrier': {
    kind: 'choice',
    id: 'ch04/wrap/q-warpbarrier',
    prompt: (
      <>
        A block of 128 threads on a device with 32-wide warps runs{' '}
        <code>if (threadIdx.x &lt; 32) {'{'} ...; __syncthreads(); {'}'}</code>. Which is true?
      </>
    ),
    options: [
      {
        label: 'It is legal: the condition is the same for every thread of a warp, so no warp diverges.',
        why: 'No warp diverges, which is section 4.5\'s question. The barrier rule is section 4.3\'s, and it is about the block.',
      },
      {
        label: 'It is legal on a device where warps run in lockstep, and undefined on every other device.',
        why: 'Lockstep is a promise about the threads inside one warp. The other three warps never reach this barrier at all.',
      },
      {
        label: 'No warp diverges, and it is still illegal: three of the four warps never reach the barrier.',
        correct: true,
      },
      {
        label: 'It is illegal only because the condition splits the first warp on a device with 64-wide warps.',
        why: 'At 64 the first warp does diverge, but the barrier is illegal at 32 already, where nothing diverges.',
      },
    ],
    explain: (
      <p>
        Two questions that are easy to merge. Does the condition split a warp? That decides passes, and here it does
        not at width 32. Does every thread of the block reach the same barrier? That decides whether the kernel is
        defined, and here 96 threads never do. NVIDIA allows a barrier in conditional code "only if the condition
        evaluates uniformly across the entire thread block". A condition can be uniform per warp and still not per
        block.
      </p>
    ),
    source: 'ch. 4 §4.3 p. 72 and §4.5 p. 81; notes/today/ch04-facts.md A6',
  },
  'ch04/wrap/q-rowmask': {
    kind: 'choice',
    id: 'ch04/wrap/q-rowmask',
    prompt: (
      <>
        An attention kernel covers a tile with blocks of 16 × 8 threads (<code>blockDim.x = 16</code>,{' '}
        <code>blockDim.y = 8</code>) and skips the tile's rows past the end of a sequence with{' '}
        <code>if (threadIdx.y &lt; K)</code>. For which K does no warp diverge, at a width of 32 and at a width of 64?
      </>
    ),
    options: [
      {
        label: 'Any K: the condition tests y, and a warp is cut along x.',
        why: 'A warp is cut from the linear index, x fastest. With rows 16 threads long, one warp spans several rows, and a test on y can split it.',
      },
      {
        label: 'K = 2',
        why: `At width 32 a warp is two rows and K = 2 is a warp boundary. At 64 a warp is four rows, and K = 2 splits the first one: ${divergentWarps(2, 64)} divergent warp.`,
      },
      { label: 'K = 4', correct: true },
      {
        label: 'K = 6',
        why: `Fine at 32, where 6 is a multiple of two rows. At 64 rows 4 to 7 are one warp and K = 6 splits it: ${divergentWarps(6, 64)} divergent warp.`,
      },
    ],
    explain: (
      <p>
        A 16-wide row puts {32 / TILE.x} rows in a 32-wide warp and {64 / TILE.x} in a 64-wide one. A row test splits
        no warp when K is a multiple of the rows per warp at every width the code will meet. K = 4 is the only option
        that is a multiple of both 2 and 4: {divergentWarps(4, 32)} divergent warps at 32 and {divergentWarps(4, 64)}{' '}
        at 64. The width belongs to the device, so a kernel that wants this property has to read it, as section 4.4
        said.
      </p>
    ),
    source: 'ch. 4 §4.4 pp. 75–76 (the cut) and §4.5 p. 81 (conditions on threadIdx); notes/today/ch04-facts.md A8, B6; counted with app/src/sim/warps.ts',
  },
  'ch04/wrap/q-pack': {
    kind: 'numeric',
    id: 'ch04/wrap/q-pack',
    prompt: (
      <>
        In section 4.6's model, a decode kernel's warps issue {BEFORE_PACK.work} instructions per load and then wait{' '}
        {BEFORE_PACK.latency} cycles, so {NEEDED_BEFORE} warps keep one issue slot busy. The weights are then stored in 8
        bits and four come in with each load, so each load now feeds {AFTER_PACK.work} instructions and the wait is the
        same. How many warps keep the slot busy now?
      </>
    ),
    answer: NEEDED_AFTER,
    unit: 'warps',
    explain: (
      <p>
        One warp uses the slot for {AFTER_PACK.work} of every {AFTER_PACK.work + AFTER_PACK.latency} cycles, so it takes{' '}
        {AFTER_PACK.work + AFTER_PACK.latency} / {AFTER_PACK.work}, rounded up, = {NEEDED_AFTER} warps, down from{' '}
        {NEEDED_BEFORE}. The wait did not get shorter. Each wait now covers more arithmetic, so fewer warps are needed to
        hide it, and the occupancy the kernel needs falls with them.
      </p>
    ),
    source: 'our model of ch. 4 §4.6, pp. 83–85 (app/src/sim/latency.ts); the 8-bit weights are ours',
  },
  'ch04/wrap/q-smallgrid': {
    kind: 'choice',
    id: 'ch04/wrap/q-smallgrid',
    prompt: (
      <>
        A decode-time kernel at batch 1 launches {SMALL_GRID.blocks} blocks of {SMALL_GRID.threadsPerBlock} threads on a
        device with {SMALL_GRID.units} SMs, where registers limit it to {SMALL_GRID.slotsBefore} blocks per SM. Its warps
        wait on memory most of the time. Someone trims its registers so that {SMALL_GRID.slotsAfter} blocks fit per SM.
        Will that help?
      </>
    ),
    options: [
      {
        label: 'Yes: twice as many blocks fit on each SM, so twice as many warps are resident to cover the waits.',
        why: `Twice as many could fit. The grid has ${SMALL_GRID.blocks} blocks, and ${CAP_BEFORE} could already run at once.`,
      },
      {
        label: `No: all ${SMALL_GRID.blocks} blocks already run at once, with room for ${CAP_BEFORE}; the new slots would stay empty.`,
        correct: true,
      },
      {
        label: 'Yes: with more blocks per SM the last wave gets shorter, so the tail effect shrinks.',
        why: `There is only one wave, and it is partial: ${SMALL_GRID.blocks} of ${CAP_BEFORE} slots.`,
      },
      {
        label: `It depends on the order in which the runtime places the blocks on the ${SMALL_GRID.units} SMs.`,
        why: `Whatever the order, at most ${SMALL_GRID.blocks} blocks exist to be placed.`,
      },
    ],
    explain: (
      <p>
        {SMALL_GRID.blocks} blocks on a device that runs {CAP_BEFORE} at once is {waves(SMALL_GRID.blocks, CAP_BEFORE)} of
        a wave; at {CAP_AFTER} it is {waves(SMALL_GRID.blocks, CAP_AFTER)}. Either way the whole device holds the grid's{' '}
        {n(WARPS_IN_GRID)} warps and no more. Occupancy is a ceiling on resident warps; the grid decides whether
        anything reaches it. A small grid that waits on memory needs more parallel work, not more room: a grid with more
        threads, for example each block's work split in two.
      </p>
    ),
    source: 'ch. 4 §4.3 p. 73 (waves) and §4.7 p. 85 (occupancy); the decode setting is ours',
  },
  'ch04/wrap/q-twodevices': {
    kind: 'choice',
    id: 'ch04/wrap/q-twodevices',
    prompt: (
      <>
        A kernel uses {KERNEL_40.regs} registers per thread and {KERNEL_40.threads}-thread blocks. It runs on the book's
        example SM ({n(BOOK.threadSlots!)} thread slots, {BOOK.blockSlots} block slots, {n(BOOK.regs!)} registers) and
        on an SM with {n(SMALLER_SM.threadSlots!)} thread slots, {SMALLER_SM.blockSlots} block slots and the same
        registers. What are its occupancies?
      </>
    ),
    options: [
      {
        label: `${pct(ON_BOOK.byWarps)}% on both: the kernel, its registers and its blocks are the same on the two SMs.`,
        why: 'Occupancy is a fraction of each SM\'s own warp slots, and the two SMs have different numbers of them.',
      },
      {
        label: `${pct(ON_BOOK.byWarps)}% on the book's example and ${pct(ON_SMALLER.byWarps)}% on the other, so the other SM keeps more warps resident.`,
        why: `Count the warps: ${ON_BOOK.residentWarps} on each.`,
      },
      {
        label: `${pct(ON_BOOK.byWarps)}% on the book's example and ${pct(ON_SMALLER.byWarps)}% on the other, with the same ${ON_BOOK.residentWarps} warps resident on each.`,
        correct: true,
      },
      {
        label: `${pct(ON_SMALLER.byWarps)}% on both: neither the thread limit nor the block limit is reached on either SM.`,
        why: `Registers are: a block takes ${n(KERNEL_40.threads * KERNEL_40.regs)} of them, so ${ON_BOOK.blocks} blocks fit, ${ON_BOOK.residentWarps} warps, which is not all of the book's ${ON_BOOK.warpSlots} warp slots.`,
      },
    ],
    explain: (
      <p>
        One block needs {KERNEL_40.threads} × {KERNEL_40.regs} = {n(KERNEL_40.threads * KERNEL_40.regs)} registers, so{' '}
        {ON_BOOK.blocks} blocks fit in {n(BOOK.regs!)} on both SMs: {ON_BOOK.residentWarps} warps. On the book's example
        that is {ON_BOOK.residentWarps} of {ON_BOOK.warpSlots} warp slots, {pct(ON_BOOK.byWarps)}%; on the other it is{' '}
        {ON_SMALLER.residentWarps} of {ON_SMALLER.warpSlots}, {pct(ON_SMALLER.byWarps)}%. The percentage describes the
        SM as much as the kernel. Compare devices by warps resident, and read every limit from the device.
      </p>
    ),
    source: "ch. 4 §4.7 pp. 85–86; per-SM limits differ across today's GPUs (notes/today/ch04-facts.md A12); computed with app/src/sim/occupancy.ts",
  },
};

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const m1 = backends.filter(
    (b) =>
      labStatus(reports[b], 'ch03/lab3_naive_gemm', 'work') === 'pass' &&
      labStatus(reports[b], 'ch04/lab1_launchconfig', 'work') === 'pass',
  );
  const where = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch04/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul>
        <li>
          A GPU is an array of SMs (compute units on AMD; Apple documents no counterpart). A grid is handed to them
          block by block, each block whole, in any order. That is why a block is the scope of a barrier and of shared
          memory, and why one grid runs unchanged on a device of any size.
        </li>
        <li>
          A barrier must be reached by every thread of the block, at the same call. A condition around it has to be
          uniform across the block, not just across a warp.
        </li>
        <li>
          A block is cut into warps from its linear thread index, x fastest. The width belongs to the device, and on
          Metal to the compiled pipeline: read it, never write 32.
        </li>
        <li>
          Threads of one warp that disagree at a branch cost extra passes. A boundary check costs little once the data
          is large. Per-thread trip counts and position-dependent masks are where divergence costs.
        </li>
        <li>
          An SM holds many more warps than it executes, so that a waiting warp costs nothing while another is ready.
          How many it needs depends on how much work a warp does between waits.
        </li>
        <li>
          Occupancy is resident warps over warp slots. Block slots, thread slots, registers and (chapter 5) shared
          memory each cap it, on each device differently, and a small change can drop it off a cliff. More of it pays
          only while there are waits left to hide.
        </li>
        <li>Ask the device: the properties struct on CUDA and HIP, the device and the compiled pipeline on Metal.</li>
      </ul>
      <Source section="4.9" pages="89–90" />

      <h2>A mixed check</h2>
      <p>
        These draw on the whole chapter and on situations the sections did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
      </p>
      <Beyond>
        <p>
          The kernel, the attention and decode settings and the 8-bit weights are ours, and the cycle counts are
          inputs to section 4.6's model, not measurements.
        </p>
      </Beyond>
      <Code
        tabs={[
          {
            label: 'CUDA (not a lab file: for this question only)',
            code: WARP_TAIL_SUM,
            status: 'front-end-checked; never run',
            tiers: { in: 'global', out: 'global', s: 'shared', v: 'shared', t: 'reg', stride: 'reg' },
          },
        ]}
      />
      <Ask q={q['ch04/wrap/q-warpsync']} label="Read the code" />
      <Ask q={q['ch04/wrap/q-warpbarrier']} />
      <Ask q={q['ch04/wrap/q-rowmask']} />
      <Ask q={q['ch04/wrap/q-pack']} />
      <Ask q={q['ch04/wrap/q-smallgrid']} />
      <Ask q={q['ch04/wrap/q-twodevices']} />

      <h2>The book's exercises</h2>
      <p>
        The chapter ends with nine exercises (pp. 90–92). Each is worked somewhere in this course; the book prints no
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
              <td>1 (a, b)</td>
              <td>Warps per block and in the grid for a given kernel</td>
              <td>a question in <a href={href('ch04', '4.4')}>4.4</a></td>
            </tr>
            <tr>
              <td>1 (c, d, e)</td>
              <td>Active and divergent warps, SIMD efficiency, loop iterations</td>
              <td>the ten-number prediction and the warp stepper in <a href={href('ch04', '4.5')}>4.5</a></td>
            </tr>
            <tr>
              <td>2, 3</td>
              <td>Threads in a grid, and warps that diverge at its boundary check</td>
              <td>two questions in <a href={href('ch04', '4.5')}>4.5</a></td>
            </tr>
            <tr>
              <td>4</td>
              <td>Time spent waiting at a barrier</td>
              <td>the barrier timeline and its prediction in <a href={href('ch04', '4.3')}>4.3</a></td>
            </tr>
            <tr>
              <td>5</td>
              <td>Leaving out the barrier in 32-thread blocks</td>
              <td>
                a question in <a href={href('ch04', '4.5')}>4.5</a>, the experiment of{' '}
                <a href={href('ch04', 'labs')}>lab 4.3</a>, and the first question above
              </td>
            </tr>
            <tr>
              <td>6</td>
              <td>The block size that puts most threads on an SM</td>
              <td>
                a question in <a href={href('ch04', '4.7')}>4.7</a>; the model tests of lab 4.4
              </td>
            </tr>
            <tr>
              <td>7, 8</td>
              <td>Which assignments fit, at what occupancy, and what limits it</td>
              <td>
                the checker at the end of <a href={href('ch04', '4.7')}>4.7</a>; the model tests of lab 4.4
              </td>
            </tr>
            <tr>
              <td>9</td>
              <td>A launch the device cannot have run</td>
              <td>a question in <a href={href('ch04', '4.8')}>4.8</a></td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Where the PDF looks wrong</h2>
      <p>
        The copy of the book this course was built from has a few apparent slips in chapter 4, and a few statements
        that have dated. They are listed so that you are not thrown by them when you read the book alongside.
      </p>
      <ul data-list="errata">
        <li>
          p. 69 says the limits on blocks per SM are discussed in "Section 4.6". They are in section 4.7 (
          <a href={href('ch04', '4.2')}>4.2</a> notes this).
        </li>
        <li>
          Fig. 4.4 (p. 72) lacks <code>__global__</code>, and the text cites its lines one higher than they are printed (
          <a href={href('ch04', '4.3')}>4.3</a>).
        </li>
        <li>
          The sidebar on pp. 78–79 points at arrows "in Fig. 4.8"; the arrows are in the sidebar's own diagram on p. 79.
        </li>
        <li>
          Fig. 4.9 draws the code after the branch as one fully active pass. The text a page later says reconvergence
          cannot be assumed from Volta on; read the figure with that.
        </li>
        <li>
          Fig. 4.10's thread labels (0, 1, 2, 4, …) disagree with the text's "threads 1 and 7". The course uses the
          printed trip counts and says so (<a href={href('ch04', '4.5')}>4.5</a>).
        </li>
        <li>
          p. 82: "one of 313" warps for 10,000 elements is one of 314 launched; 313 hold an element. And the "200 × 150"
          picture has 75 or 100 divergent warps of 1040, depending on which number is the width, not the printed 80.
          With 200 rows the share is above the book's "under 8%" (<a href={href('ch04', '4.5')}>4.5</a>).
        </li>
        <li>p. 81 calls chapter 3's <code>vecAddKernel</code> by another name, <code>addVecKernel</code>.</li>
        <li>
          Occupancy is defined with warps on p. 85 and with threads on p. 90. They agree when warps are full; the
          course uses warps (<a href={href('ch04', '4.7')}>4.7</a>).
        </li>
        <li>p. 86 says the shared-memory analysis is "in 5."; chapter 5 is meant.</li>
        <li>
          "SIMD efficiency" and "active" warps appear only in exercise 1 and are defined nowhere; section 4.5 gives the
          definitions it uses.
        </li>
        <li>
          p. 88: the device loop compares an <code>unsigned int</code> with the <code>int</code> count and checks no
          return code (<a href={href('ch04', '4.8')}>4.8</a>, lab 4.1).
        </li>
        <li>
          p. 89 describes <code>regsPerBlock</code> as the registers of an SM. NVIDIA documents it as the limit for one
          block; the SM's pool is <code>regsPerMultiprocessor</code>. The same list's <code>clockRate</code> was removed
          from the struct in CUDA 13.0 (<a href={href('ch04', '4.8')}>4.8</a>).
        </li>
        <li>
          p. 88 expects two or more CUDA devices in a typical PC because of integrated GPUs. No NVIDIA source we found
          supports that reason, and integrated GPUs from other vendors are not CUDA devices.
        </li>
      </ul>
      <Aside>
        <p>
          These are our readings of one PDF, checked against current vendor documentation on 2026-11-20
          (notes/today/ch04-facts.md). If your copy differs, trust your copy.
        </p>
      </Aside>
      <p>
        Two points about the book's example device stay open. Its four processing blocks of 32 per SM could not be
        confirmed either way, and the lessons present them as the book's figure. And the book gives no split of its 132
        SMs among 8 GPCs, and NVIDIA does not publish one.
      </p>
      <p>
        Some of the book is left out on purpose, so that you can tell an omission from a gap when you read alongside:
        the list of compilers in the sidebar on intrinsic functions (p. 70), Flynn's taxonomy in the sidebar on warps
        and SIMD hardware (pp. 78–79), the post-office analogy for latency tolerance (p. 83) and the hotel analogy for
        querying a device (p. 87). The lessons make the same points with their own pictures, and nothing later depends
        on these.
      </p>

      <h2>The project so far</h2>
      <p>
        Lab 3.3's naive GEMM and lab 4.1's launch, sized from what the device reports, are milestone M1 of the
        through-line project: a linear layer with nothing about the device written into its source.{' '}
        <span data-m1={m1.join(',') || 'none'}>
          {backends.length === 0
            ? 'No lab results are recorded on this machine yet.'
            : m1.length
              ? `Your own labs 3.3 and 4.1 both pass on ${m1.map(where).join(' and ')} on this machine.`
              : 'Your own labs 3.3 and 4.1 do not both pass on any backend of this machine yet.'}
        </span>{' '}
        See the <a href={href('ch04', 'labs')}>labs</a>.
      </p>
      <p>
        Chapter 5 turns to the memory on the chip. Registers and shared memory are where a kernel keeps the data it
        reuses, and they are also two of this chapter's occupancy budgets: section 5.6 adds shared memory to the
        arithmetic you did here. Its first move is the one this chapter's labs keep pointing at: the naive kernel waits
        on global memory, and tiling makes it wait less. Start at <a href={href('ch05', 'start')}>chapter 5</a>.
      </p>
    </>
  );
}
