import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Source } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';

// A kernel of our own, in the spirit of the book's exercise 10 (p. 120): each block
// transposes one B x B tile of a matrix in place, through shared memory.
const TRANSPOSE = `#define B 8   // tile width; the matrix width is a multiple of B

__global__ void transposeTiles(float* A, int width) {
    __shared__ float tile[B][B];
    int idx = (blockIdx.y * B + threadIdx.y) * width + blockIdx.x * B + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = A[idx];
    A[idx] = tile[threadIdx.x][threadIdx.y];
}`;

export const questionsWrap: Record<string, Question> = {
  'ch05/wrap/q-transpose': {
    kind: 'choice',
    id: 'ch05/wrap/q-transpose',
    prompt: (
      <>
        The kernel above is launched with <code>B × B</code> threads per block. For which values of <code>B</code> is
        it guaranteed to produce the transposed tiles?
      </>
    ),
    options: [
      { label: 'Any B.', why: 'Each thread reads a slot that a different thread writes. What makes that write happen first?' },
      { label: 'Any B up to the width of a warp.', why: 'Lockstep execution inside one warp may hide the problem, but the language does not promise it, and a block with B = 8 already has 64 threads.' },
      { label: 'Only B = 1.', correct: true },
      { label: 'No B at all.', why: 'With one thread per block the only slot read is the one the thread itself just wrote.' },
    ],
    explain: (
      <p>
        Thread (x, y) reads <code>tile[x][y]</code>, which thread (y, x) writes. Nothing orders the two, so the read
        can see uninitialised memory: a read-after-write dependence with no barrier. Only when the thread reads its own
        slot (x = y, which for every thread means B = 1) is it safe. The fix is one <code>__syncthreads()</code>{' '}
        between the two statements. No second barrier is needed, because the tile is never overwritten.
      </p>
    ),
    source: 'after ch. 5 exercise 10, pp. 119–120; dependence vocabulary from §5.4, pp. 110–111',
  },
  'ch05/wrap/q-flopb': {
    kind: 'numeric',
    id: 'ch05/wrap/q-flopb',
    prompt: (
      <>
        Per thread, a kernel loads six floats from global memory, loads two more from shared memory, performs nine
        multiplications and eight additions, and stores two floats to global memory. What is its ratio in FLOP per byte
        of global-memory traffic?
      </>
    ),
    answer: 0.53,
    tol: 0.03,
    unit: 'FLOP/B',
    explain: (
      <p>
        17 operations. Global traffic: six loads and two stores of 4 bytes = 32 bytes. 17 / 32 = 0.53 FLOP/B. The two
        shared-memory loads cost nothing against global bandwidth, which is the whole reason to put reused data there.
      </p>
    ),
    source: 'method of ch. 5 §5.1, pp. 94–97; compare exercise 11(f), p. 121',
  },
  'ch05/wrap/q-notile': {
    kind: 'choice',
    id: 'ch05/wrap/q-notile',
    prompt: (
      <>
        You are asked to speed up a kernel that computes <code>y[i] = a * x[i] + y[i]</code> for a very long vector.
        Would loading <code>x</code> and <code>y</code> into shared memory tiles help?
      </>
    ),
    options: [
      { label: 'Yes: shared memory is faster than global memory, so any kernel gains from using it.',
        why: 'The data still has to come from global memory once. Copying it to shared memory first adds work unless something reuses it.' },
      { label: 'No: each element is read by exactly one thread, exactly once, so there is no repeated fetch to remove.', correct: true },
      { label: 'Yes, but only if the tile is at least as wide as a warp.', why: 'Tile size does not create reuse that the computation does not have.' },
    ],
    explain: (
      <p>
        Tiling removes <em>redundant</em> traffic. It needs data that several threads of a block want, or that one
        thread wants several times. Here every byte is touched once: 2 FLOP for 12 bytes, 0.17 FLOP/B, and that is
        already the minimum traffic for this computation. The kernel is memory-bound by nature, and the useful question
        becomes how close to the speed of light it runs.
      </p>
    ),
    source: 'ch. 5 exercise 1, p. 119, and §5.1, p. 96',
  },
  'ch05/wrap/q-measured': {
    kind: 'choice',
    id: 'ch05/wrap/q-measured',
    prompt: (
      <>
        On some device the naive matmul kernel measures eight times <em>above</em> the ceiling that 0.25 FLOP/B and the
        measured bandwidth predict. Before running anything else, what should you expect from tiling with T = 16 there?
      </>
    ),
    options: [
      { label: 'About 16× faster, as the model says.', why: 'The model\'s 16× is a reduction in requested traffic. Much of that traffic was evidently not reaching global memory in the first place.' },
      { label: 'Noticeably less than 16×: caches were already serving most of the repeated reads that tiling removes.', correct: true },
      { label: 'No change at all.', why: 'Tiling still removes work (and can also improve access patterns), so some gain is likely; it is just not the model\'s factor.' },
    ],
    explain: (
      <p>
        A naive kernel running far above its model ceiling is telling you its effective ratio is already much higher
        than 0.25, because on-chip caches absorb repeated reads. Tiling makes that reuse explicit and reliable, but it
        can only save traffic that was really happening. On the machine this course was built on, the gain at T = 16
        was under 2×. Predict, then measure.
      </p>
    ),
    source: 'ch. 5 §5.1, p. 97 (caches raise the real ratio); measurements from lab 5.1 and 5.2',
  },
  'ch05/wrap/q-llm': {
    kind: 'choice',
    id: 'ch05/wrap/q-llm',
    prompt: (
      <>
        Token-by-token generation at batch size 1 spends most of its time in matrix-vector products. Which change
        raises the operations-per-byte ratio of those products?
      </>
    ),
    options: [
      { label: 'A larger tile width.', why: 'Each weight is used once per token; there is no repeated fetch for a tile to share.' },
      { label: 'More threads per block.', why: 'That changes how the work is divided, not how many bytes each operation needs.' },
      { label: 'Storing the weights in 8 bits instead of 32.', correct: true },
      { label: 'Adding a second barrier.', why: 'Barriers are about correctness, not traffic.' },
    ],
    explain: (
      <p>
        Two operations per weight, one fetch of that weight: the ratio is 2 divided by the bytes per weight. Float32
        gives 0.5 FLOP/B, int8 gives 2. That is a fourfold rise in the ceiling for a memory-bound kernel, before any
        arithmetic gets cheaper. The other lever is batching, which lets one fetched weight serve several rows.
      </p>
    ),
    source: 'our analysis with the §5.1 method; the book makes the batching argument in §20.6, p. 505',
  },
};

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = realBackends(reports);
  const m2 = backends.filter((b) => labStatus(reports[b], 'ch05/lab3_gemm', 'work') === 'pass');
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch05/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul>
        <li>
          A kernel has a ratio, operations per byte of global-memory traffic, and a device has a threshold, peak
          throughput over peak bandwidth. Below the threshold, memory decides how fast you go.
        </li>
        <li>
          The model counts what a kernel asks for. Caches decide how much of that reaches global memory, so the model
          gives direction and a measurement gives magnitude.
        </li>
        <li>
          Data can live in registers (one thread), shared memory (one block), global or constant memory (everyone).
          Faster memories are smaller, and using them costs occupancy.
        </li>
        <li>
          Tiling moves reused data into a faster memory. For matmul with <code>T × T</code> tiles it cuts requested
          traffic by a factor of <code>T</code>, and it needs two barriers per phase: one for a true dependence, one
          for a false one.
        </li>
        <li>
          Every memory access gets its own bounds check. A thread that owns no output may still have to load for its
          block.
        </li>
        <li>Capacities are properties of a device. Query them; do not assume them.</li>
      </ul>
      <Source section="5.7" pages="118–119" />

      <h2>A mixed check</h2>
      <p>
        These draw on the whole chapter and on situations the sections did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly on the first recorded try.` : ''}
      </p>
      <Code tabs={[{ label: 'CUDA (not a lab file: for this question only)', code: TRANSPOSE, tiers: { A: 'global', tile: 'shared', idx: 'reg' } }]} />
      <Ask q={questionsWrap['ch05/wrap/q-transpose']} label="Find the defect" />
      <Ask q={questionsWrap['ch05/wrap/q-flopb']} />
      <Ask q={questionsWrap['ch05/wrap/q-notile']} />
      <Ask q={questionsWrap['ch05/wrap/q-measured']} />
      <Ask q={questionsWrap['ch05/wrap/q-llm']} />

      <h2>The book's exercises</h2>
      <p>
        The chapter ends with twelve exercises (pp. 119–121). Each is covered somewhere in this course; the book's own
        wording is worth doing as well.
      </p>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th>Exercise</th><th>About</th><th>Where it is worked here</th></tr>
        </thead>
        <tbody>
          <tr><td>1</td><td>Shared memory for matrix addition</td><td><a href={href('ch05', '5.3')}>5.3</a>, and the saxpy question above</td></tr>
          <tr><td>2</td><td>8×8 with 2×2 and 4×4 tiles</td><td>the "who reads what" explorer in <a href={href('ch05', '5.3')}>5.3</a></td></tr>
          <tr><td>3</td><td>Forgetting a barrier</td><td>stepper and outcome table in <a href={href('ch05', '5.4')}>5.4</a>; lab 5.4</td></tr>
          <tr><td>4</td><td>Shared memory versus registers</td><td><a href={href('ch05', '5.2')}>5.2</a></td></tr>
          <tr><td>5, 8</td><td>Traffic reduction with tiles</td><td><a href={href('ch05', '5.3')}>5.3</a></td></tr>
          <tr><td>6, 7</td><td>How many copies of a variable</td><td><a href={href('ch05', '5.2')}>5.2</a></td></tr>
          <tr><td>9</td><td>Compute- or memory-bound on two devices</td><td><a href={href('ch05', '5.1')}>5.1</a></td></tr>
          <tr><td>10</td><td>A tile transpose with a defect</td><td>the first question above (our own kernel in the same spirit)</td></tr>
          <tr><td>11</td><td>Copies, shared bytes and ratio of a kernel</td><td><a href={href('ch05', '5.2')}>5.2</a>, and the ratio question above</td></tr>
          <tr><td>12</td><td>Occupancy under three budgets</td><td>the budget calculator in <a href={href('ch05', '5.6')}>5.6</a></td></tr>
        </tbody>
      </table>

      <h2>Where the PDF looks wrong</h2>
      <p>
        This edition is new and the copy this course was built from has a few apparent slips in chapter 5. They are
        listed so that you are not thrown by them when you read the book alongside.
      </p>
      <ul>
        <li>p. 97 gives the example device's threshold as "25.1 FLOP/B or 20.0 FLOP/B". Everywhere else it is 20.0, which is what the two peaks divide to.</li>
        <li>Fig. 5.8 numbers the phases 0 and 1; the text beside it calls them 1 and 2. The code counts from 0, and so does this course.</li>
        <li>Fig. 5.14 (dynamically sized shared memory) has several errors as printed; <a href={href('ch05', '5.6')}>section 5.6</a> goes through them and lab 5.5 has a corrected, tested version.</li>
        <li>p. 113 says thread (1,0) of block (1,1) must load M[2][1] although it owns no output. By the book's own (row, column) naming that thread would load M[3][0]; the thread that loads M[2][1] is (0,1). The point it illustrates stands.</li>
        <li>p. 116 describes <code>sharedMemPerBlock</code> as the shared memory of an SM. The CUDA documentation defines it per block.</li>
        <li>Exercise 12 states shared-memory use "per SM" for a kernel; per block is almost certainly meant.</li>
      </ul>
      <Aside>
        <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
      </Aside>

      <h2>The project so far</h2>
      <p>
        Lab 5.3's kernel, a tiled matrix multiplication for any shapes, is milestone M2 of the through-line
        project: it is the computation behind every linear layer of the small transformer you will end up running.{' '}
        {backends.length === 0
          ? 'No lab results are recorded on this machine yet.'
          : m2.length
            ? `Your own version passes on ${m2.join(' and ')} on this machine.`
            : 'Your own version has not passed on this machine yet.'}{' '}
        See the <a href={href('ch05', 'labs')}>labs</a>.
      </p>
      <p>
        Chapter 6 picks up a detail this chapter stepped around. The naive kernel reads M along memory and N across it.
        Those two patterns do not cost the same, and the reason, how a warp's memory requests are merged or not, is the
        next large lever after tiling.
      </p>
    </>
  );
}
