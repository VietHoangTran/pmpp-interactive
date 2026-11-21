import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { stripWorst } from '../../sim/gemmbanks';
import { SCHEDULES, run } from '../../sim/gemmpipe';
import { readsPerFma } from '../../sim/gemmreads';
import { intensity } from '../../sim/gemmreuse';

// A kernel of our own, for one question: section 15.8's toy loop with its two calls
// swapped. Checked copy: labs/ch15/snippets/wrap_load_first.cu (Wrap.snippets.test.ts
// ties the two together and pins the numbers below to the models).
const LOAD_FIRST = `// Section 15.8's pipelined toy (s158_pipeline_toy.cu) with the two calls in the loop
// swapped: the loads of tile + 1 come before mm on tile. For a question on the
// chapter's summary page; not built or run.
#define THREADS 4

__device__ __forceinline__ void loadTile(float* buf, const float* tiles, unsigned int tile) {
    buf[threadIdx.x] = tiles[tile * THREADS + threadIdx.x];
}

__device__ __forceinline__ void mm(const float* buf, float& sum) {
    for (unsigned int j = 0; j < THREADS; ++j) sum += buf[j];
}

__global__ void loadFirstToy(const float* tiles, float* sums, unsigned int numTiles) {
    __shared__ float buf[2][THREADS];
    float sum = 0.0f;
    if (numTiles == 0) {
        sums[threadIdx.x] = sum;
        return;
    }
    loadTile(buf[0], tiles, 0);
    __syncthreads();
    for (unsigned int tile = 0; tile < numTiles - 1; ++tile) {
        loadTile(buf[(tile + 1) & 1], tiles, tile + 1);
        mm(buf[tile & 1], sum);
        __syncthreads();
    }
    mm(buf[(numTiles - 1) & 1], sum);
    sums[threadIdx.x] = sum;
}`;

// ---------------------------------------------------------------- the numbers the questions take from the models
/** A batch-4 step through a linear layer: a 4-row output tile, 128 wide (src/sim/gemmreuse.ts). */
export const DECODE4 = { m: 4, n: 128, fpb: intensity(4, 128), bound: 0.5 * 4 };
/** Leading dimensions of A_s for bK = 16, §15.5 layout, 32 banks (src/sim/gemmbanks.ts). The A-strip
 *  conflict depends only on the row stride, so the model's eight steps along i are enough. */
export const BK16 = Object.fromEntries(
  [16, 17, 20, 24].map((ld) => [ld, stripWorst({ layout: 'remap', strip: 'A', ld, banks: 32, W: 32 }).degree]),
) as Record<16 | 17 | 20 | 24, number>;
/** The load-first loop under every thread order, 1 to 6 tile pairs (src/sim/gemmpipe.ts). */
export const LOAD_FIRST_OK = SCHEDULES.every((s) =>
  [1, 2, 3, 4, 5, 6].every((n) => run({ threads: 4, tiles: n, variant: 'load-first' }, s).correct),
);
/** 8 x 8 against 4 x 4 thread tiles, 256 threads (src/sim/gemmreads.ts, gemmreuse.ts). */
export const SMALLER = {
  reads8: readsPerFma({ tM: 8, tN: 8, bK: 8 }, 'v2'),
  reads4: readsPerFma({ tM: 4, tN: 4, bK: 8 }, 'v2'),
  fpb128: intensity(128, 128),
  fpb64: intensity(64, 64),
};
const r2 = (v: number) => Math.round(v * 100) / 100;

export const questionsWrap: Record<string, Question> = {
  'ch15/wrap/q-decode4': {
    kind: 'numeric',
    id: 'ch15/wrap/q-decode4',
    prompt: (
      <>
        Speculative decoding checks 4 drafted tokens at once, so every linear layer runs at batch 4. In this
        chapter&rsquo;s kernel a block&rsquo;s output tile is then only 4 rows tall, the batch, and 128 wide. What is that
        tile&rsquo;s intensity with 4-byte elements?
      </>
    ),
    answer: r2(DECODE4.fpb),
    tol: 0.02,
    unit: 'FLOP/B',
    explain: (
      <p>
        0.5 · 4 · 128 / (4 + 128) = {r2(DECODE4.fpb)} FLOP/B, and 124 of the block&rsquo;s 128 rows compute nothing. A
        wider tile does not rescue it: 0.5·m·n/(m + n) stays below 0.5·m, here {DECODE4.bound}, however large n grows.
        The weights are read once either way, so while the layer is memory-bound, checking four tokens takes about as
        long as generating one: that is the bet speculative decoding makes. Section{' '}
        <a href={href('ch15', '15.2')}>15.2</a> has the formula; the speculative-decoding setting and that reading of it
        are ours.
      </p>
    ),
    source: 'formula of ch. 15 §15.2, p. 351; the setting is ours',
  },
  'ch15/wrap/q-bk16': {
    kind: 'choice',
    id: 'ch15/wrap/q-bk16',
    prompt: (
      <>
        You double bK to 16 in lab 15.4&rsquo;s kernel (the section 15.5 layout, 32-lane warps). Which leading dimension
        of <code>A_s</code> makes the A-strip reads conflict-free on 32 banks of 4 bytes?
      </>
    ),
    options: [
      {
        label: '16: the tile is 16 wide, so its rows are already 16 words apart.',
        why: `Lane rows 4 rows apart start 64 words apart, which is 0 mod 32: one bank for all eight, a ${BK16[16]}-way conflict.`,
      },
      { label: '17.', correct: true },
      {
        label: '20: four words of padding, like the project’s B_s.',
        why: `Rows 4 apart start 80 words apart, 16 mod 32: the eight lane rows share two banks, a ${BK16[20]}-way conflict. The B_s pitch of lab 15.6 solves a different access.`,
      },
      {
        label: '24: half a row more, so rows land on different halves of the banks.',
        why: `Rows 4 apart start 96 words apart, 0 mod 32: a ${BK16[24]}-way conflict, as bad as no padding.`,
      },
    ],
    explain: (
      <p>
        The eight lane rows of a warp read rows 4 apart, so lane row g reads word 4·g·ld plus a constant. The eight land
        in eight different banks only when 4·g·ld mod 32 takes eight different values for g = 0 to 7, which happens
        exactly when ld is odd (the lanes within one lane row read the same word, a broadcast). With bK = 16 the smallest odd choice is 17 ({BK16[17] === 1 ? 'conflict-free' : `${BK16[17]}-way`}{' '}
        in the model), the same one column of padding as the book&rsquo;s 9 for bK = 8. Section{' '}
        <a href={href('ch15', '15.6')}>15.6</a>&rsquo;s explorer gives the same rule for any width; on 64 banks the
        model asks only that ld not be divisible by 4.
      </p>
    ),
    source: 'rule of ch. 15 §15.6, pp. 361–363, and ch15-facts E1 (our arithmetic with NVIDIA’s bank rule); bK = 16 is ours',
  },
  'ch15/wrap/q-load-first': {
    kind: 'choice',
    id: 'ch15/wrap/q-load-first',
    prompt: (
      <>
        The kernel above is section 15.8&rsquo;s toy with the two calls in the loop swapped: the loads of pair{' '}
        <code>tile + 1</code> now come before <code>mm</code> on pair <code>tile</code>, and the one barrier stays at the
        end of the iteration. Is it still correct?
      </>
    ),
    options: [
      {
        label: 'No: a fast thread\'s loads now come before its mm, so they can overwrite the buffer a slower thread\'s mm is reading.',
        why: 'The loads write buf[(tile + 1) & 1] and mm reads buf[tile & 1]: different buffers in every iteration.',
      },
      {
        label: 'Yes: the loads fill the buffer freed before the last barrier, and mm reads the one that barrier completed.',
        correct: true,
      },
      {
        label: 'No: it needs a second barrier between the loads and mm.',
        why: 'A barrier there would order the loads before mm, but mm does not read anything these loads write.',
      },
      {
        label: 'Only when numTiles is even, so that the epilogue reads the buffer the last load filled.',
        why: 'The parity alternates every iteration, whatever the count; the epilogue reads (numTiles - 1) & 1 either way.',
      },
    ],
    explain: (
      <p>
        In iteration <code>tile</code> the loads write <code>buf[(tile + 1) &amp; 1]</code>, which was last read by{' '}
        <code>mm</code> in iteration <code>tile - 1</code>, before that iteration&rsquo;s barrier; <code>mm</code> reads{' '}
        <code>buf[tile &amp; 1]</code>, which was filled before the same barrier. Neither order inside the iteration
        matters to correctness. The thread-level model of section <a href={href('ch15', '15.8')}>15.8</a>{' '}
        {LOAD_FIRST_OK ? 'finds no wrong sum' : 'finds a wrong sum'} under any of its thread orders for 1 to 6 tiles.
        What the order changes is the code the compiler is handed, which is why lab 15.5 asks you to measure both.
      </p>
    ),
    source: 'Fig. 15.14, p. 367; the variant and the toy are ours (src/sim/gemmpipe.ts)',
  },
  'ch15/wrap/q-epilogue': {
    kind: 'choice',
    id: 'ch15/wrap/q-epilogue',
    prompt: (
      <>
        Your model&rsquo;s layers compute y = gelu(x Wᵀ + b). Today you run M3, then a second kernel that reads Y, adds b,
        applies GELU and writes Y again. Where can the bias and the activation go so that Y is written once and never
        read back?
      </>
    ),
    options: [
      {
        label: 'Into loadTile: add b to each element of the X tile as it arrives in shared memory.',
        why: 'The bias is added once per output. Adding it to an input element changes every product that element enters.',
      },
      {
        label: 'Into mm: add b[col] to each accumulator at every step of the inner loop over k.',
        why: 'That adds the bias at every step along the inner dimension: K times instead of once.',
      },
      {
        label: 'Into writeTile: add b[col] and apply GELU in registers before the store.',
        correct: true,
      },
      {
        label: 'Nowhere in M3: fusing an epilogue needs a library such as cuBLASLt.',
        why: 'Your kernel holds every output in registers until writeTile. That is the easiest place in any GEMM to fuse, yours included.',
      },
    ],
    explain: (
      <p>
        Until <code>writeTile</code>, each output lives only in its thread&rsquo;s <code>C_r</code>. Adding{' '}
        <code>b[col]</code> there (read once per column the thread owns) and applying GELU costs a few instructions per
        output and saves a whole second pass over Y: one read and one write of batch × out floats. The book&rsquo;s own
        GEMM form, D = αA × B + βC, is the same idea for a bias with α = β = 1. This is the &ldquo;fused epilogue&rdquo;
        reason to write your own kernel that section <a href={href('ch15', '15.9')}>15.9</a> gives; CUTLASS calls the
        store phase the epilogue too.
      </p>
    ),
    source: 'GEMM form, ch. 15 §15.1, p. 350; writeTile, Fig. 15.7, p. 357; the fusion is ours; CUTLASS epilogue, ch15-facts A12',
  },
  'ch15/wrap/q-triage': {
    kind: 'choice',
    id: 'ch15/wrap/q-triage',
    prompt: (
      <>
        A profiler on your M3 kernel reports 12.5% achieved occupancy, the fused multiply-add pipe busy most of the
        time, and almost no shared-memory bank conflicts. A colleague proposes 4 × 4 thread tiles, so that the registers
        allow four blocks per SM. What do you expect?
      </>
    ),
    options: [
      {
        label: 'Faster: four blocks per SM give four times the warps, and four times the latency is hidden.',
        why: 'Latency hiding helps when the units wait. Here the pipe is already busy, and NVIDIA’s triage guide calls extra warps counter-productive in that case.',
      },
      {
        label: 'No better, probably worse: the units are already busy, and smaller tiles reuse each loaded value less.',
        correct: true,
      },
      {
        label: 'Faster, because each block then needs less shared memory, so more of them fit.',
        why: 'Shared memory was never the limit (section 15.7). Per output, smaller tiles load more, not less.',
      },
      {
        label: 'No change: in a compute-bound kernel, occupancy has no effect on speed.',
        why: 'Too strong. The proposal changes more than occupancy: it changes how much each loaded value is reused.',
      },
    ],
    explain: (
      <p>
        With 256 threads, 4 × 4 thread tiles make a 64 × 64 block tile: the intensity falls from{' '}
        {r2(SMALLER.fpb128)} to {r2(SMALLER.fpb64)} FLOP/B, and each thread&rsquo;s shared reads per multiply-add rise
        from {r2(SMALLER.reads8)} to {r2(SMALLER.reads4)}. Section <a href={href('ch15', '15.7')}>15.7</a>&rsquo;s trade
        runs the other way on purpose. Raise occupancy when the units wait for data, not when they are busy.
      </p>
    ),
    source: 'ch. 15 §15.7, pp. 363–364; Nsight Compute triage guide, ch15-facts A3; the profile is made up',
  },
};

// ---------------------------------------------------------------- the page
export function Wrap() {
  const { progress } = useProgress();
  const { reports, loading } = useReports();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const m3 = backends.filter((b) => labStatus(reports[b], 'ch15/lab6_linear', 'work') === 'pass');
  const name = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch15/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <p>
        The chapter applied chapter 6&rsquo;s optimisations to one kernel, and each one removed the bottleneck the
        previous one exposed. The parts to keep:
      </p>
      <Source section="15.10" pages="370" />
      <ul>
        <li>
          A block that owns an m × n output tile does 0.5·m·n/(m + n) FLOP per byte of input it loads. The depth of a
          step cancels, so k is free to be small, and nothing about the tile helps when m is 1: a batch-1 linear layer
          is memory-bound however it is written.
        </li>
        <li>
          Large tiles mean coarse threads: 256 threads own a 128 × 128 tile, 64 outputs each. Those outputs stay in
          registers only if every index is a constant, so every loop that touches them is fully unrolled and every
          helper inlined.
        </li>
        <li>
          Tiling works at every level where two memories differ in speed. Register tiling is a loop interchange: walk
          the inner dimension outermost, hold one strip of A and one of B in registers, and do a rank-1 update.
        </li>
        <li>
          Each fix can move the bottleneck. Re-mapping threads to 4 × 4 sub-tiles makes the stores coalesce and raises a
          shared-memory conflict; one padding column (any odd leading dimension, on 32 banks) removes it, with one line
          of code the book does not mention.
        </li>
        <li>
          This kernel runs at low occupancy on purpose: registers are the limit, and reuse is worth more than warps.
          Wider accesses, full unrolling and a pipelined loop stand in for the warps it gives up.
        </li>
        <li>
          Two buffers remove the false dependence; rotating the loop puts the loads of the next pair and the arithmetic
          on this one in the same iteration, under one barrier. The overlap itself is the compiler&rsquo;s to make, or
          the hardware&rsquo;s with asynchronous copies where a backend has them.
        </li>
        <li>
          Call the library unless it lacks your operation, and measure against it either way. In FP32 the gap is
          technique (asynchronous copies, per-shape kernel selection, deeper tuning; on AMD CDNA possibly FP32 matrix
          instructions), and what each backend lets your own code use differs. Tensor cores enter only when you opt
          into a lower precision (section 15.9).
        </li>
      </ul>

      <h2>A mixed check</h2>
      <p>
        These use the whole chapter on situations the sections did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
      </p>
      <Beyond>
        <p>The settings, the toy kernel and the profile in these questions are ours, picked for the questions.</p>
      </Beyond>
      <Ask q={q['ch15/wrap/q-decode4']} />
      <Ask q={q['ch15/wrap/q-bk16']} />
      <Code
        tabs={[
          {
            label: 'CUDA (not a lab file: for the next question only)',
            code: LOAD_FIRST,
            status: 'front-end-checked; never run',
            tiers: { tiles: 'global', sums: 'global', buf: 'shared', sum: 'reg' },
          },
        ]}
        caption={<>From <code>labs/ch15/snippets/wrap_load_first.cu</code>.</>}
      />
      <Ask q={q['ch15/wrap/q-load-first']} />
      <Ask q={q['ch15/wrap/q-epilogue']} />
      <Ask q={q['ch15/wrap/q-triage']} />

      <h2>The book&rsquo;s exercises</h2>
      <p>
        The chapter ends with three coding exercises (p. 370), and the book gives no solution for any of them. Each is
        worked here, as a lab with tests and a benchmark.
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
              <td>Vector loads in <code>loadTile</code></td>
              <td>
                the loadTile stepper&rsquo;s vector mode in <a href={href('ch15', '15.3')}>15.3</a>, which flags
                misaligned addresses; lab 15.4 on the <a href={href('ch15', 'labs')}>labs page</a>, and lab 15.6 for the
                W tile
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td>Vector stores in <code>writeTile</code></td>
              <td>
                the store-pattern explorer in <a href={href('ch15', '15.5')}>15.5</a>, which shows why vector stores
                alone leave gaps; lab 15.3
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td>The kernel with section 15.5&rsquo;s four 4 × 4 sub-tiles per thread</td>
              <td>
                the lane picker and the narrated animation in <a href={href('ch15', '15.5')}>15.5</a>, for 32- and 64-wide
                warps; lab 15.3, then labs 15.4 to 15.6, which keep the layout
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Where the PDF looks wrong</h2>
      <p>
        The copy this course was built from has a few apparent slips and loose ends in chapter 15. They are listed so
        that you are not thrown by them when you read the book alongside.
      </p>
      <Aside>
        <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
      </Aside>
      <ul data-list="errata">
        <li>
          pp. 362–363: padding <code>A_s</code> changes its declaration and the leading dimension passed to{' '}
          <code>loadTile</code> and <code>mm</code>, but the thread&rsquo;s start offset <code>&amp;A_s[tRow*bK]</code>{' '}
          must change to bK + 1 as well, and the book does not say so. Left as printed, threads read the wrong rows. See{' '}
          <a href={href('ch15', '15.6')}>15.6</a>; lab 15.4&rsquo;s tests catch it.
        </li>
        <li>
          p. 363: &ldquo;B needs no change&rdquo; holds in section 15.5&rsquo;s layout only. In the layout of sections
          15.3 and 15.4 the B-strip read is a 4-way conflict on 32 banks, which is why the labs remap before they pad.
        </li>
        <li>
          p. 365 says the first instructions &ldquo;load the input tiles A0 and B0 from shared memory&rdquo;; from global
          memory into shared memory is meant, as p. 366 says.
        </li>
        <li>
          pp. 352 and 354: the prose describes a block of 16 × 16 threads, but Fig. 15.3 computes both tile coordinates
          from <code>threadIdx.x</code> alone, which needs a one-dimensional block of 256. Section{' '}
          <a href={href('ch15', '15.3')}>15.3</a> shows what a 16 × 16 launch would do.
        </li>
        <li>
          Fig. 15.14 (p. 367) is a schematic: its arguments are elided and in another order than Figs. 15.5 and 15.6,
          its buffers and <code>numTiles</code> are not declared, and its 2D pointer types do not match Fig. 15.3&rsquo;s
          one-dimensional tiles. Lab 15.5 has a compiled, tested version.
        </li>
        <li>
          Fig. 15.3 declares the shared tiles inside the loop body (lines 23 and 24). The labs declare them once, at the
          top of the kernel.
        </li>
        <li>
          p. 362: the banks quoted for the &ldquo;second elements&rdquo; (1 unpadded; 1, 5, …, 29 padded) fit column 1 of
          the same rows, the next step along the inner dimension, not the next row. The same page invites a check of
          &ldquo;all the 16 elements accessed by each 4-thread group&rdquo; without saying which 16.
        </li>
        <li>
          p. 364 says vector loads and stores were seen &ldquo;in Section 15.3&rdquo;; that section only names them as
          exercises, and no vectorised code appears in the chapter.
        </li>
        <li>
          pp. 359 and 364 credit register tiling with raising arithmetic intensity. By section 15.2&rsquo;s count the
          intensity comes from the size of the output tile, which registers hold; register tiling of the inputs cuts
          shared-memory reads.
        </li>
        <li>
          p. 353 says a local array is placed &ldquo;in the global memory&rdquo; by default; chapter 5 and NVIDIA call
          the place local memory, physically in device memory.
        </li>
        <li>
          Small slips: Fig. 15.6&rsquo;s first line is clipped before the comma after <code>unsigned int k</code> (Fig.
          15.9 shows it); Fig. 15.7 shows no opening brace; p. 363 writes <code>bk</code> where the code has{' '}
          <code>bK</code>; p. 369 has &ldquo;multidimensional tenors&rdquo; for tensors.
        </li>
        <li>
          The chapter reports no measurement for any step and shows no host code or launch configuration. The labs fill
          both in.
        </li>
      </ul>
      <p>
        Several dated statements in sections 15.7 and 15.9 have moved since the book was written, and one could not be
        confirmed. Sections <a href={href('ch15', '15.7')}>15.7</a> and <a href={href('ch15', '15.9')}>15.9</a> go
        through them; in short:
      </p>
      <Today checked="2026-11-20 (ch15-facts A3, A5 to A9)">
        <p>
          2048 resident threads per SM holds only on some GPUs (1536 on RTX 30 to 50). CUDA 13 targets compute
          capability 7.5 and later. WMMA is still supported; WGMMA is Hopper-only; TMEM is not on RTX 50. No NVIDIA page
          says nvcc fuses a vector load and store into LDGSTS. cuTile is now CUDA Tile.
        </p>
      </Today>

      <h2>What this course shortened</h2>
      <p>
        Every section of the chapter has its own page here. Kept to a sentence or a pointer: the graphics example of
        section 15.1 (p. 350); the α and β of the full GEMM form, which the book also leaves out (p. 350); storage-order
        combinations other than the book&rsquo;s row-major pair and the project&rsquo;s (out, in) weights; and the GPU-
        and cluster-level tiles of the hierarchy (pp. 352, 359–360). Explained but not built: warp specialisation (p.
        368), which needs producer and consumer synchronisation the book does not show; asynchronous copies (p. 369),
        which only CUDA documents in a usable form and the emulators do not model; tensor cores and their interfaces
        (pp. 368–369), which differ by vendor (section{' '}
        <a href={href('x2', 'X2.5')}>X2.5</a> names their integer forms; no lab uses them); and cuDNN, named only (section <a href={href('ch19', '19.3')}>19.3</a> lowers convolution to GEMM,
        and <a href={href('ch19', '19.4')}>19.4</a> is about cuDNN).
      </p>

      <h2>The project so far</h2>
      <p>
        Lab 15.6 is milestone M3: the project&rsquo;s linear layer with the kernel of sections 15.3 to 15.8 inside,
        behind the same <code>launchLinear(X, W, Y, batch, in, out)</code> as M2, and benchmarked against M2 and the
        vendor library.{' '}
        <span data-m3={loading ? 'loading' : m3.join(',') || 'none'}>
          {loading
            ? 'Reading reports.'
            : backends.length === 0
              ? 'No lab results are recorded on this machine yet.'
              : m3.length
                ? <>Your own <code>launchLinear</code> passes on {m3.map(name).join(' and ')} on this machine.</>
                : <>Your own <code>launchLinear</code> has not passed on this machine yet.</>}
        </span>{' '}
        See the <a href={href('ch15', 'labs')}>labs</a>. Which of M2 and M3 to call at a given batch size is the
        benchmark&rsquo;s answer for your device, not the chapter&rsquo;s.
      </p>
      <p>
        The book&rsquo;s Part 3 begins with <a href={href('ch16')}>chapter 16</a>, on computations whose cells wait for
        earlier cells. For the project, the next user of this kernel is <a href={href('ch20')}>chapter 20</a>: the Q Kᵀ
        and P V products of attention are section 15.1&rsquo;s GEMMs, and milestone M6 builds on them.
      </p>
    </>
  );
}
