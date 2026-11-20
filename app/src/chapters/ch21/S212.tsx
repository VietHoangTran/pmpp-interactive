import { Ask, Predict, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { href } from '../../lib/router';
import { PairsExplorer } from './PairsExplorer';
import { ScatterGatherScene } from './ScatterGatherScene';

// Section 21.2 of the app covers the book's §§21.2 to 21.4 (pp. 515–522): scatter versus
// gather, thread coarsening, and coalescing-aware ownership. Code shown here is the part
// below the marker line of each file in labs/ch21/snippets (snippets.test.ts ties them).

const SCATTER = `__constant__ float atoms[CHUNK_SIZE * 4];  // x, y, z, charge of each atom in the chunk

__global__ void cenergy(float* energygrid, dim3 grid, float gridspacing, float z, int numatoms) {
    int n = (blockIdx.x * blockDim.x + threadIdx.x) * 4;  // this thread's atom
    if (n >= numatoms * 4) return;                        // added: Fig. 21.5 has no bound
    int k = z / gridspacing;                              // which slice
    float dz = z - atoms[n + 2];  // the same for every point of the slice
    float dz2 = dz * dz;
    float charge = atoms[n + 3];
    for (int j = 0; j < (int)grid.y; j++) {
        float y = gridspacing * (float)j;
        float dy = y - atoms[n + 1];  // the same for every point of a row
        float dy2 = dy * dy;
        int row = grid.x * grid.y * k + grid.x * j;
        for (int i = 0; i < (int)grid.x; i++) {
            float x = gridspacing * (float)i;
            float dx = x - atoms[n];
            // every thread adds to every point: the update must be atomic
            atomicAdd(&energygrid[row + i], charge / sqrtf(dx * dx + dy2 + dz2));
        }
    }
}`;

const GATHER = `__constant__ float atoms[CHUNK_SIZE * 4];  // x, y, z, charge of each atom in the chunk

__global__ void cenergy(float* energygrid, dim3 grid, float gridspacing, float z, int numatoms) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= (int)grid.x || j >= (int)grid.y) return;  // added: Fig. 21.6 relies on padding
    int atomarrdim = numatoms * 4;
    int k = z / gridspacing;
    float y = gridspacing * (float)j;  // recomputed by every thread
    float x = gridspacing * (float)i;
    float energy = 0.0f;
    for (int n = 0; n < atomarrdim; n += 4) {  // every lane reads the same atom
        float dx = x - atoms[n];
        float dy = y - atoms[n + 1];
        float dz = z - atoms[n + 2];
        energy += atoms[n + 3] / sqrtf(dx * dx + dy * dy + dz * dz);
    }
    // one plain write, to a point no other thread touches
    energygrid[grid.x * grid.y * k + grid.x * j + i] += energy;
}`;

const PRINTED = `__constant__ float atoms[CHUNK_SIZE*4];
#define COARSEN_FACTOR 4
void __global__ cenergy(float *energygrid, dim3 grid, float gridspacing,
                        float z, int numatoms) {
    int i = blockIdx.x * blockDim.x*COARSEN_FACTOR + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int atomarrdim = numatoms * 4;
    int k = z / gridspacing;
    float y = gridspacing * (float) j;
    float x = gridspacing * (float) i;
    float energy0 = 0.0f;
    float energy1 = 0.0f;
    float energy2 = 0.0f;
    float energy3 = 0.0f;
    // add up the contributions of every atom
    for (int n=0; n<atomarrdim; n+=4) {
        float dx0 = x - atoms[n  ];
        float dx1 = dx0 +   gridspacing;
        float dx2 = dx0 + 2*gridspacing;
        float dx3 = dx0 + 3*gridspacing;
        float dy = y - atoms[n+1];
        float dz = z - atoms[n+2];
        float dysqdzsq = dy*dy + dz*dz;
        float charge = atoms[n+3];
        energy0 += charge / sqrtf(dx0*dx0 + dysqdzsq);
        energy1 += charge / sqrtf(dx1*dx1 + dysqdzsq);
        energy2 += charge / sqrtf(dx2*dx2 + dysqdzsq);
        energy3 += charge / sqrtf(dx3*dx3 + dysqdzsq);
    }
    energygrid[grid.x*grid.y*k + grid.x*j + i  ] += energy0;
    energygrid[grid.x*grid.y*k + grid.x*j + i+1] += energy1;
    energygrid[grid.x*grid.y*k + grid.x*j + i+2] += energy2;
    energygrid[grid.x*grid.y*k + grid.x*j + i+3] += energy3;
}`;

const ADJACENT = `__constant__ float atoms[CHUNK_SIZE * 4];
#define COARSEN_FACTOR 4

__global__ void cenergy(float* energygrid, dim3 grid, float gridspacing, float z, int numatoms) {
    // corrected: the thread index is scaled, so thread t owns points 4t .. 4t + 3
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * COARSEN_FACTOR;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= (int)grid.y) return;
    int atomarrdim = numatoms * 4;
    int k = z / gridspacing;
    float y = gridspacing * (float)j;
    float x = gridspacing * (float)i;
    float energy0 = 0.0f, energy1 = 0.0f, energy2 = 0.0f, energy3 = 0.0f;
    for (int n = 0; n < atomarrdim; n += 4) {
        float dx0 = x - atoms[n];  // one load of the atom serves four points
        float dx1 = dx0 + gridspacing;
        float dx2 = dx0 + 2 * gridspacing;
        float dx3 = dx0 + 3 * gridspacing;
        float dy = y - atoms[n + 1];
        float dz = z - atoms[n + 2];
        float dysqdzsq = dy * dy + dz * dz;  // shared by the four points of the row
        float charge = atoms[n + 3];
        energy0 += charge / sqrtf(dx0 * dx0 + dysqdzsq);
        energy1 += charge / sqrtf(dx1 * dx1 + dysqdzsq);
        energy2 += charge / sqrtf(dx2 * dx2 + dysqdzsq);
        energy3 += charge / sqrtf(dx3 * dx3 + dysqdzsq);
    }
    float* row = energygrid + grid.x * grid.y * k + grid.x * j;
    // four write statements; in each, neighbouring lanes are 4 floats apart
    if (i < (int)grid.x) row[i] += energy0;
    if (i + 1 < (int)grid.x) row[i + 1] += energy1;
    if (i + 2 < (int)grid.x) row[i + 2] += energy2;
    if (i + 3 < (int)grid.x) row[i + 3] += energy3;
}`;

const STRIDED = `__constant__ float atoms[CHUNK_SIZE * 4];
#define COARSEN_FACTOR 4

__global__ void cenergy(float* energygrid, dim3 grid, float gridspacing, float z, int numatoms) {
    int i = blockIdx.x * blockDim.x * COARSEN_FACTOR + threadIdx.x;  // line 05, as printed
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= (int)grid.y) return;
    int atomarrdim = numatoms * 4;
    int k = z / gridspacing;
    float y = gridspacing * (float)j;
    float x = gridspacing * (float)i;
    float energy0 = 0.0f, energy1 = 0.0f, energy2 = 0.0f, energy3 = 0.0f;
    for (int n = 0; n < atomarrdim; n += 4) {
        float dx0 = x - atoms[n];
        // the thread's points are blockDim.x apart, and so are their distances
        float dx1 = dx0 + blockDim.x * gridspacing;
        float dx2 = dx0 + 2 * blockDim.x * gridspacing;
        float dx3 = dx0 + 3 * blockDim.x * gridspacing;
        float dy = y - atoms[n + 1];
        float dz = z - atoms[n + 2];
        float dysqdzsq = dy * dy + dz * dz;
        float charge = atoms[n + 3];
        energy0 += charge / sqrtf(dx0 * dx0 + dysqdzsq);
        energy1 += charge / sqrtf(dx1 * dx1 + dysqdzsq);
        energy2 += charge / sqrtf(dx2 * dx2 + dysqdzsq);
        energy3 += charge / sqrtf(dx3 * dx3 + dysqdzsq);
    }
    float* row = energygrid + grid.x * grid.y * k + grid.x * j;
    int B = blockDim.x;
    // four write statements; in each, neighbouring lanes write neighbouring floats
    if (i < (int)grid.x) row[i] += energy0;
    if (i + B < (int)grid.x) row[i + B] += energy1;
    if (i + 2 * B < (int)grid.x) row[i + 2 * B] += energy2;
    if (i + 3 * B < (int)grid.x) row[i + 3 * B] += energy3;
}`;

/** Lines of PRINTED the erratum hunt refers to (1-based). */
export const PRINTED_LINES = { index: 5, writes: [30, 31, 32, 33] };

const STATUS = 'front-end-checked; emulator-checked (python3 labs/ch21/snippets/check.py); unverified on CUDA or HIP hardware';
const TIERS: Record<string, Tier> = {
  atoms: 'const', energygrid: 'global', row: 'global',
  energy: 'reg', energy0: 'reg', energy1: 'reg', energy2: 'reg', energy3: 'reg', charge: 'reg', dysqdzsq: 'reg',
  dx: 'reg', dy: 'reg', dz: 'reg', dx0: 'reg', dx1: 'reg', dx2: 'reg', dx3: 'reg', dy2: 'reg', dz2: 'reg',
  i: 'reg', j: 'reg', k: 'reg', n: 'reg', x: 'reg', y: 'reg', B: 'reg', atomarrdim: 'reg',
};

export const questionsS212: Record<string, Question> = {
  'ch21/21.2/q-why-no-atomic': {
    kind: 'choice',
    id: 'ch21/21.2/q-why-no-atomic',
    prompt: <>Every thread of the gather kernel reads every atom of the chunk. Why does it need no atomic operation?</>,
    options: [
      {
        label: 'Reads of shared data never conflict; only several writes to one location do, and each gather thread writes one point that no other thread writes.',
        correct: true,
      },
      { label: 'Because the atoms sit in constant memory, which kernels cannot write, so nothing about them can race.', why: 'That protects the atoms. The question is about the grid, which every launch updates.' },
      { label: 'Because the host launches the kernel once per chunk, so the updates of different threads never overlap in time.', why: 'One launch per chunk orders the chunks. Inside one launch the threads still run together.' },
      { label: 'Because a += on a float in global memory is a single indivisible instruction on current GPUs.', why: 'It is a load, an add and a store. Two threads can interleave between them.' },
    ],
    explain: (
      <p>
        A race needs two threads touching one location with at least one of them writing. All the sharing in the gather
        kernel is reading: every lane reads the same atom. The writing is private: thread (i, j) is the only one that ever
        writes point (i, j). Each launch adds one chunk's contribution with a plain += because no other thread touches that
        point during the launch.
      </p>
    ),
    source: 'ch. 21 §21.2, pp. 518–519',
  },
  'ch21/21.2/q-dilemma': {
    kind: 'choice',
    id: 'ch21/21.2/q-dilemma',
    prompt: (
      <>
        The gather kernel recomputes <code>y</code> in every thread, work that the sequential Fig. 21.4 had hoisted out of
        its inner loop. Why accept that?
      </>
    ),
    options: [
      {
        label: 'Hoisting needed the atom loop outermost, which makes threads share outputs and forces atomics; recomputing y costs a few instructions, and coarsening wins most of the reuse back inside one thread.',
        correct: true,
      },
      { label: 'Arithmetic on a GPU is free compared with memory traffic, so redundant work of any amount costs nothing at all.', why: 'Not free: the book calls this a trade-off between the amount of calculation and the degree of parallelism.' },
      { label: 'Recomputing y is needed for correctness: with y hoisted the kernel would add the atoms in the wrong order.', why: 'Both loop orders give the same sums (up to float rounding); the interchange is legal because the iterations are independent.' },
      { label: 'It is never actually recomputed: the compiler notices the repetition and hoists y out of the threads again.', why: 'A compiler hoists within one thread. Other threads are other executions of the code.' },
    ],
    explain: (
      <p>
        The optimised sequential code bought its savings by putting the atoms outermost, and that order is exactly what
        turns a parallel version into a scatter. The book's way out: parallelise the less optimised order first, accept a
        little redundant arithmetic per thread, then recover the reuse with thread coarsening, as the rest of this page does.
      </p>
    ),
    source: 'ch. 21 §21.2, p. 518 (the dilemma) and pp. 518–519',
  },
  'ch21/21.2/q-deterministic': {
    kind: 'choice',
    id: 'ch21/21.2/q-deterministic',
    prompt: (
      <>
        With <code>torch.use_deterministic_algorithms(True)</code>, the backward pass of <code>nn.Embedding</code> runs
        on a CUDA tensor but raises an error on an MPS tensor. Why?
      </>
    ),
    options: [
      {
        label: 'The CUDA path sorts the indices and sums each token\'s rows with one writer in a fixed order; the MPS kernel adds into the gradient with float atomics.',
        correct: true,
      },
      { label: 'Metal has no float atomics at all, so the MPS backend cannot run this backward pass in any mode.', why: 'It has them (atomic_float, Apple7 and later), and the MPS kernel uses them; that is why it is not deterministic.' },
      { label: 'CUDA guarantees the order of atomic additions to one address, so its atomic path is always deterministic.', why: 'NVIDIA documents that order as undefined. PyTorch skips its fused atomic path in deterministic mode for that reason.' },
      { label: 'The MPS backend computes the gradient in half precision, and half-precision sums cannot be made repeatable.', why: 'Precision is not the issue; the order of the additions is.' },
    ],
    explain: (
      <p>
        One framework, two choices: binned gather on CUDA and ROCm, atomic scatter on MPS. The scatter is fast to write and
        needs no sort, but the order in which atomics land is undefined, and float addition is not associative, so the bits
        can change from run to run. Deterministic mode asks for the gather.
      </p>
    ),
    source: 'ours, not the book: notes/today/ch21-facts.md D1, D2; infra A1',
  },
  'ch21/21.2/q-ex2': {
    kind: 'choice',
    id: 'ch21/21.2/q-ex2',
    prompt: (
      <>
        Exercise 2. Coarsening by eight cuts the constant-memory loads per atom from 32 (eight Fig. 21.6 threads) to 4.
        Which per-point work stays exactly as it was?
      </>
    ),
    options: [
      { label: 'One division and one square root per point per atom, with the dx² term, the additions inside the root and the accumulate.', correct: true },
      { label: 'The subtraction that forms dy, which every one of the eight points of the row still needs to compute.', why: 'dy is the same for the whole row; the coarsened thread computes it once per atom, not eight times.' },
      { label: 'The multiplication dy·dy, because each of the eight points still squares its own copy of dy.', why: 'It is folded into dysqdzsq once per atom.' },
      { label: 'The loop branch, because each of the eight points still runs its own loop over the atoms.', why: 'One thread runs one loop: one branch per atom instead of eight.' },
    ],
    explain: (
      <>
        <p>
          Counted from the listing, per atom: eight Fig. 21.6 threads make 32 loads, 24 subtractions, 24 multiplications, 24
          additions, 8 divisions, 8 square roots and 8 loop branches. One thread coarsened by eight makes 4 loads, 3
          subtractions, 10 multiplications (16 if the m·gridspacing products stay in the loop), 24 additions, 8 divisions, 8
          square roots and 1 branch. Set F = 8 in the operation table above to see it.
        </p>
        <p>What coarsening removes is what the points share. Every point still needs its own distance and its own 1/r.</p>
      </>
    ),
    source: 'ch. 21 Exercise 2, p. 528; notes/today/ch21-facts.md E1',
  },
  'ch21/21.2/q-printed': {
    kind: 'choice',
    id: 'ch21/21.2/q-printed',
    prompt: (
      <>
        Fig. 21.8 as printed computes <code>i = blockIdx.x*blockDim.x*COARSEN_FACTOR + threadIdx.x</code> and writes{' '}
        <code>i</code>, <code>i+1</code>, <code>i+2</code>, <code>i+3</code>. With <code>blockDim.x = 32</code>, which of
        its 128 points does block 0 write?
      </>
    ),
    options: [
      { label: 'Points 0 to 34: most of them (3 to 31) four times with a plain +=, and points 35 to 127 never.', correct: true },
      { label: 'Points 0 to 127, each exactly once, four adjacent points per thread.', why: 'That is the intended ownership, and it needs threadIdx.x multiplied by 4.' },
      { label: 'Points 0 to 31, each once: one point per thread, as in Fig. 21.6.', why: 'Each thread writes four points, i to i+3.' },
      { label: 'Points 0 to 127, each four times, so atomics would fix it.', why: 'Thread 31 starts at 31 and stops at 34; nothing reaches beyond.' },
    ],
    explain: (
      <p>
        Thread t starts at t, one point after its neighbour, and writes four points, so the block covers 0 to 31 + 3 = 34.
        Interior points are written by four threads, each adding the full potential, with a non-atomic +=. The other 93
        points of the block's stretch are never touched. Each thread's distances are right for the points it writes; the
        ownership is wrong. The text on p. 521 describes threads four points apart, which is what scaling threadIdx.x by
        the factor gives.
      </p>
    ),
    source: 'ch. 21 Fig. 21.8, p. 520, against p. 521; notes/today/ch21-facts.md E2',
  },
  'ch21/21.2/q-ex3': {
    kind: 'choice',
    id: 'ch21/21.2/q-ex3',
    prompt: (
      <>
        Exercise 3, made concrete. Fig. 21.10 with <code>blockDim.x = 128</code> on a slice 1,024 points wide. You raise
        the coarsening factor from 4 to 16. What do you lose?
      </>
    ),
    options: [
      {
        label: 'A block now spans 2,048 x positions, so half of every thread\'s 16 points fall past the row: 16 accumulators in registers for 8 points of work, and one block per row.',
        correct: true,
      },
      { label: 'Nothing: coarsening only ever removes work, so a larger factor is always at least as fast as a smaller one.', why: 'It removes redundant loads, not idle slots; the extra points past the row are pure cost.' },
      { label: 'The writes stop being coalesced, because each thread\'s points are now further apart in memory.', why: 'The stride is still blockDim.x: each write statement of a warp stays one contiguous run.' },
      { label: 'The kernel now needs atomics, because a thread\'s 16 points overlap the points of the next block.', why: 'Each point still has exactly one owner; the next block starts after this block\'s stretch.' },
    ],
    explain: (
      <p>
        The two costs the book names for more work per thread are both here. Registers: sixteen accumulators and sixteen
        distances per thread, which can lower how many warps fit (p. 521). And granularity: a block's stretch is factor ×
        blockDim.x points, so a row that is not a multiple of it leaves threads idle or doing work for points that do not
        exist, the "padding waste" of Fig. 21.9. With only one block per row there is also less parallelism to go round.
      </p>
    ),
    source: 'ch. 21 Exercise 3, p. 528; p. 521; Fig. 21.9',
  },
  'ch21/21.2/q-wavefront': {
    kind: 'choice',
    id: 'ch21/21.2/q-wavefront',
    prompt: <>On an AMD Instinct GPU a wavefront has 64 lanes. Is Fig. 21.10's stride still right there?</>,
    options: [
      {
        label: 'Yes: the stride is blockDim.x, not the warp width. With blockDim.x of 64 or more each statement of a wavefront is one run of 256 bytes.',
        correct: true,
      },
      { label: 'No: on AMD the stride has to be 64, the wavefront width, instead of blockDim.x.', why: 'blockDim.x already is the stride, and it is chosen as a multiple of the wavefront width.' },
      { label: 'No: AMD coalesces per half wavefront, so the stride has to be 32 floats instead.', why: 'Half-warps are a 2007 NVIDIA unit. AMD documents grouping a wavefront\'s addresses into cache lines.' },
      { label: 'It does not matter: AMD GPUs do not coalesce memory accesses, so every pattern costs the same.', why: 'AMD documents exactly that grouping, into 64- or 128-byte lines.' },
    ],
    explain: (
      <p>
        Lane t of a statement writes base + t + m·blockDim.x, so neighbouring lanes always write neighbouring floats, whatever
        the group width. What the width changes is how far one statement reaches: 128 bytes for 32 lanes, 256 bytes for
        64. With <code>blockDim.x = 32</code> a 64-lane wavefront holds two rows of the block, so each statement becomes two
        runs, a row apart. Set the write-statement view to 64 lanes and blockDim.x = 32 to see it.
      </p>
    ),
    source: 'notes/today/ch21-facts.md B1, B3, E2; ch05-facts B5',
  },
  'ch21/21.2/q-metal-edge': {
    kind: 'choice',
    id: 'ch21/21.2/q-metal-edge',
    prompt: (
      <>
        A Metal port of Fig. 21.10 is launched with <code>dispatchThreads</code> and reads its stride from{' '}
        <code>[[threads_per_threadgroup]]</code>. The slice is not a multiple of the threadgroup width. What goes wrong?
      </>
    ),
    options: [
      {
        label: 'The last threadgroup in x is smaller and reports its actual size, so its threads stride by a smaller number and own the wrong points.',
        correct: true,
      },
      { label: 'Nothing: Metal pads the edge threadgroup with idle threads, so every group reports the same width.', why: 'That is dispatchThreadgroups. dispatchThreads creates a smaller edge group instead.' },
      { label: 'The kernel fails to compile, because the stride must be a constant known when the pipeline is built.', why: 'A stride read from an attribute is an ordinary runtime value.' },
      { label: 'Only coalescing suffers: the edge group\'s writes are scattered, but every point is still right.', why: 'The edge group\'s points and distances use the wrong stride, so the values land in the wrong places.' },
    ],
    explain: (
      <p>
        The MSL specification says <code>[[threads_per_threadgroup]]</code> is the actual size of the threadgroup running
        the kernel, which can be smaller at the edge of a <code>dispatchThreads</code> grid, while{' '}
        <code>[[dispatch_threads_per_threadgroup]]</code> is the size given at dispatch. Take the stride from the second,
        or launch whole threadgroups with <code>dispatchThreadgroups</code>, and check each of the thread's points against
        the row.
      </p>
    ),
    source: 'notes/today/ch05-facts.md C4 (MSL spec 5.2.3.6), C5; ch21-facts C4',
  },
  'ch21/21.2/q-ex4': {
    kind: 'choice',
    id: 'ch21/21.2/q-ex4',
    prompt: (
      <>
        Exercise 4. You keep adjacent ownership and replace the four scalar <code>+=</code> with one <code>float4</code>{' '}
        load, four adds and one <code>float4</code> store per thread. What must be true of the slice?
      </>
    ),
    options: [
      { label: 'Every row must start on a 16-byte boundary: grid.x a multiple of 4, or rows allocated with a pitch.', correct: true },
      { label: 'Nothing: any float buffer can be read and written as float4, whatever the address.', why: 'A 16-byte access needs a 16-byte-aligned address; NVIDIA\'s guidance says so.' },
      { label: 'blockDim.x must be a multiple of 4, so that every thread gets a whole float4.', why: 'Alignment is about addresses, not about how many threads a block has.' },
      { label: 'The block must first stage its outputs in shared memory, and store them from there.', why: 'Not needed: each thread already owns four adjacent floats in registers.' },
    ],
    explain: (
      <p>
        Thread t's four points start at element 4t of its stretch, and the stretch starts a whole row further on for every
        row. If <code>grid.x</code> is not a multiple of 4, the second row's float4s are misaligned. With aligned rows a warp
        then writes 512 contiguous bytes in one request: 16 sectors, NVIDIA's optimum for 128-bit accesses. On Metal you
        declare the buffer <code>device float4*</code> instead of casting a <code>float*</code>.
      </p>
    ),
    source: 'ch. 21 Exercise 4, p. 528; p. 521; notes/today/ch21-facts.md A9, C4',
  },
};

/** The coarsened explorer opens only after the erratum question has an answer, so the explorer does not give it away. */
function AfterPrinted() {
  const { progress } = useProgress();
  if (!progress.answers['ch21/21.2/q-printed'])
    return (
      <p className="narrate" data-gate="q-printed">
        Answer the erratum question above first. The explorer that follows runs the printed line 05 and would give it away.
      </p>
    );
  return (
    <PairsExplorer
      title="Who writes where: coarsened kernels"
      task="Start with the printed line 05 and look for writers per point above 1 and hatched points. Then switch to the two corrected mappings."
      views={['mapping']}
      initialKernel="printed"
      foot="Model: src/sim/dcs.ts. Coarsened kernels draw each thread's points in the same shade; the table below the drawing counts work per atom from the listings."
    />
  );
}

export function S212() {
  return (
    <>
      <p>
        The chapter's computation is a sum over all pairs. At every point of a 3D grid, the electrostatic potential is the
        sum over all atoms of the atom's charge divided by its distance to the point. The book handles one 2D slice of the
        grid at a time, so the work per slice is the number of atoms times the number of points in the slice. Strip away the
        chemistry and this page is about one question that every all-pairs kernel faces: which thread writes which output.
      </p>
      <Source section="21.1" pages="514–515" figs="Fig. 21.2" />

      <h2>Two loop orders</h2>
      <p>
        The book's plain C version (Fig. 21.3) loops over the rows of the slice, then over the points of a row, then over the
        atoms, and writes each point once. Its optimised version (Fig. 21.4) interchanges the loops so that the atoms are
        outermost. That buys real savings: the z distance is the same for every point of the slice, so it is squared once
        per atom, and the y distance is the same along a row, so it is squared once per row. The interchange is legal because
        every iteration is independent: the same terms get added, in a different order. The price is that each point is now
        updated with <code>+=</code> once per atom instead of being written once.
      </p>
      <Source section="21.2" pages="515–517" figs="Figs. 21.3, 21.4" />

      <h2>Scatter: one thread per atom</h2>
      <p>
        Parallelise the optimised version the obvious way and each thread takes one iteration of the outer loop: one atom.
        It then runs both grid loops itself and <em>scatters</em> its atom's contribution to every point of the slice. Every
        point receives an update from every thread, so every update has to be atomic.
      </p>
      <Source section="21.2" pages="517–518" figs="Fig. 21.5" />
      <Code
        tabs={[{ label: 'scatter (Fig. 21.5)', code: SCATTER, tiers: TIERS, status: STATUS }]}
        caption={
          <>
            Fig. 21.5 as the lesson shows it, from <code>labs/ch21/snippets/scatter.cu</code>. The listing has no atom count
            and no bound; this one adds both, and computes the slice index the way Fig. 21.6 does. The legacy{' '}
            <code>atomicAdd</code> is kept as printed. <TierLegend tiers={['const', 'global', 'reg']} />
          </>
        }
      />
      <p>
        Where do the atoms come from? Constant memory is small, so the host feeds them in chunks of{' '}
        <code>CHUNK_SIZE</code>. For each slice it copies the slice to the device, then for each chunk copies the chunk into
        the <code>__constant__</code> array and launches the kernel, which adds that chunk's contribution with{' '}
        <code>+=</code>; after the last chunk the slice goes back to the host.
      </p>
      <Source section="21.2" pages="517" />
      <Today checked="2026-11-20 (notes/today/ch21-facts.md A5, B1, C1, E3; infra-facts A5, B5, C6)">
        <p>
          NVIDIA: 64 KB of <code>__constant__</code> holds 4,096 atoms, but the constant cache's working set is 8 KB per SM:
          512 atoms. HIP keeps the spelling (<code>hipMemcpyToSymbol</code>) but documents no size, and its scalar cache
          helps only reads that are the same across the wavefront. Metal has no symbol copy: a host-filled buffer taken as{' '}
          <code>constant float4*</code>, no documented size; <code>setBytes</code> takes under 4 KB, which is 255 atoms.
        </p>
      </Today>
      <Predict
        id="ch21/21.2/p-lockstep"
        prompt={
          <>
            A warp of the scatter kernel runs the two grid loops in lockstep. In one <code>atomicAdd</code> instruction, how
            many distinct grid addresses do its 32 lanes target?
          </>
        }
        unit="addresses"
      >
        {(v) => (
          <p>
            <b>One.</b> Every thread walks the points in the same order, so at each instruction all the lanes add to the same
            point: a full-warp collision on one address at every step. The lanes differ only in the value they add, and the
            hardware applies same-address atomics one after another.{' '}
            {v === 1 ? 'That is what you had.' : v === 32 ? 'Thirty-two would be one point per lane. That is the gather kernel, below: there the threads own the points.' : ''}
          </p>
        )}
      </Predict>
      <ScatterGatherScene />
      <p>
        The animation shows two costs that the book does not spell out. First, the order in which a queue of same-address
        atomics lands is undefined, and float addition is not associative, so the scatter kernel's sums can differ from one
        run to the next in the last bits. Each gather thread adds its atoms in the loop's order, so its bits never change.
        Second, the scatter kernel's lanes each read a different atom from constant memory, which is the case NVIDIA says
        is serialised: one access per distinct address.
      </p>
      <Beyond title="Not in the book: two more costs">
        <p>
          Both follow from vendor documentation (infra-facts A1: same-address atomics serialised in undefined order;
          ch10-facts A9: float addition not associative; ch07-facts A3: one constant-cache access per distinct address).
          No vendor gives a speed ratio for scatter against gather (ch21-facts A6).
        </p>
      </Beyond>
      <Today checked="2026-11-20 (notes/today/infra-facts.md A1, B1, C1; ch21-facts C2, ch09-facts C4)">
        <p>
          NVIDIA now calls float <code>atomicAdd</code> a legacy function and recommends <code>cuda::atomic_ref</code>.
          HIP compiles it as a safe operation, possibly a compare-and-swap loop, unless you build with{' '}
          <code>-munsafe-fp-atomics</code>, and lists no hardware float atomic add on RDNA1 and CDNA1. Metal needs the grid
          typed <code>device atomic_float*</code> and <code>atomic_fetch_add_explicit(…, memory_order_relaxed)</code>, on
          Apple7 and later; Apple documents neither where atomics execute nor what they cost.
        </p>
      </Today>

      <h2>Gather: one thread per point</h2>
      <p>
        The alternative is to give each thread one point and let it <em>gather</em> the contributions of all atoms. That
        needs the loop order of the plain Fig. 21.3, and one change: to give each thread one iteration of the two outer
        loops, they must be perfectly nested, so the y coordinate moves inside and every thread computes it again. The book
        calls this a trade of calculation for parallelism. In return each thread makes one plain <code>+=</code> per launch,
        to a point no other thread touches, and at every step all the lanes of a warp read the same atom: the constant
        cache's best case, one address served to the whole warp.
      </p>
      <Source section="21.2" pages="518–519" figs="Fig. 21.6" />
      <Code
        tabs={[{ label: 'gather (Fig. 21.6)', code: GATHER, tiers: TIERS, status: STATUS }]}
        caption={
          <>
            Fig. 21.6 from <code>labs/ch21/snippets/gather.cu</code>, with a bound check on <code>i</code> and{' '}
            <code>j</code>. The printed kernel has none and relies on a slice padded to whole blocks (the "padding waste" of
            Fig. 21.9).
          </>
        }
      />
      <p>
        In the explorer, start with the scatter kernel. Every point has as many writers as there are atoms; tick the plain{' '}
        <code>+=</code> box to see what losing the atomic does. Then switch to gather, click a point and read its writer's
        index arithmetic. Drag the atoms to move them.
      </p>
      <Simplification>
        <p>
          A 2D slice with unit spacing; the atoms sit a fixed height above it so no distance is zero. Same-address atomics
          land in one fixed order here (on a GPU it is undefined). Without the atomic, every lane of a warp is taken to load
          the same old value and the highest lane's store to survive; which lane wins on a GPU is not defined. No time, no
          caches.
        </p>
      </Simplification>
      <PairsExplorer
        title="Who writes where: scatter and gather"
        task="Count the writers per point in scatter, then in gather. Watch the constant-address count too."
        views={['mapping']}
        initialKernel="scatter"
        foot="Model: src/sim/dcs.ts. Points are shaded by the thread that writes them; ×n marks a point with n writers; hatched points are never written. Results are float32 in each kernel's own order, against a float64 direct sum."
      />
      <Ask q={questionsS212['ch21/21.2/q-why-no-atomic']} />
      <p>
        This is a dilemma the book returns to: the optimised sequential code is often harder to parallelise than the plain
        one. Its savings came from the loop order that turns into a scatter. The gather kernel gives them up, and each thread
        does more arithmetic than it would have. The risk is that the slower threads eat into what parallelism gained.
      </p>
      <Source section="21.2" pages="518" />
      <Ask q={questionsS212['ch21/21.2/q-dilemma']} />
      <p>
        You have met this choice before if you have trained a language model. The backward pass of an embedding lookup adds
        each token's gradient row into the weight row of its token id, and many tokens share an id: the scatter of Fig. 21.5,
        with token ids in place of grid points.
      </p>
      <Beyond>
        <p>
          PyTorch on CUDA and ROCm sorts the indices (which bins rows by token id) and sums each bin with one writer; it uses
          a fused atomic path only when that helps and deterministic mode is off. PyTorch on MPS scatters with float atomics
          and raises an error in deterministic mode. llama.cpp's <code>get_rows_back</code> (CUDA, HIP; training only) is
          Fig. 21.6's direct-summation gather with an equality test in place of 1/r; we found no Metal version
          (ch21-facts D1 to D3).
        </p>
      </Beyond>
      <Ask q={questionsS212['ch21/21.2/q-deterministic']} />

      <h2>Coarsening: one load, several points</h2>
      <p>
        The gather kernel still makes four constant-memory loads per atom for each point. The book points out that those
        load instructions use hardware that could otherwise raise floating-point throughput, and they cost energy. Four neighbouring points of a row share their y and z distances and the atom's
        charge, and differ only in x (Fig. 21.7). So fuse four threads into one: the thread loads each atom once, keeps it in
        registers, and computes four points.
      </p>
      <Source section="21.3" pages="519–520" figs="Fig. 21.7" />
      <Predict
        id="ch21/21.2/p-loads"
        prompt={
          <>
            After coarsening by four, how many constant-memory loads does one thread make per atom for its four points?
            (Four Fig. 21.6 threads make 16.)
          </>
        }
        unit="loads"
      >
        {(v) => (
          <p>
            <b>Four:</b> x, y, z and the charge, once each. dy, dz and dy² + dz² are also computed once and kept in
            registers. {v === 4 ? 'That is what you had.' : v === 1 ? 'One would be the whole atom in a single load; the listings read four floats, one at a time.' : ''}
          </p>
        )}
      </Predict>
      <p>
        Before you look at the explorer, read the coarsened kernel as the book prints it. Each thread keeps four accumulators,
        computes <code>dx1</code> to <code>dx3</code> from <code>dx0</code>, and writes four neighbouring points on lines 30
        to 33. Then look hard at line 05.
      </p>
      <Source section="21.3" pages="520" figs="Fig. 21.8" />
      <Code
        tabs={[{ label: 'Fig. 21.8 as printed: do not use', code: PRINTED, tiers: TIERS, status: 'front-end-checked; emulator-checked to fail as described' }]}
        highlight={[PRINTED_LINES.index, ...PRINTED_LINES.writes]}
        caption={
          <>
            The 34 code lines of Fig. 21.8 with the book's numbering, from <code>labs/ch21/snippets/coarsen_printed.cu</code>;
            the comment on line 15 is reworded. Kept for the erratum hunt only.
          </>
        }
      />
      <Ask q={questionsS212['ch21/21.2/q-printed']} label="Erratum hunt" />
      <AfterPrinted />
      <p>
        The fix is one multiplication: scale the thread index by the factor, so thread t starts at point 4t of its block's
        stretch. That is the ownership the text describes (p. 521), and it is the version lab 21.1 provides as its baseline.
        Switch the explorer to "coarsened, adjacent": every point has one writer and the sums match.
      </p>
      <Code
        tabs={[{ label: 'Fig. 21.8, corrected', code: ADJACENT, tiers: TIERS, status: STATUS }]}
        caption={
          <>
            From <code>labs/ch21/snippets/coarsen_adjacent.cu</code>: line 05 scales the thread index; one bound check per
            point replaces the padding.
          </>
        }
      />
      <p>
        How much does it save? The book counts 16 loads and 48 operations down to 4 loads and 24 operations. The listings
        do not quite say that. Counted from the code, four Fig. 21.6 threads make 40 arithmetic operations per atom and the
        coarsened thread 27, or 25 if the 2·gridspacing and 3·gridspacing products are hoisted. The book counts 12
        divisions where the code has 4, and counts no square roots at all. The operation table in the explorer above counts
        them for any factor.
      </p>
      <Source section="21.3" pages="520–521" />
      <Aside>
        <p>
          The book's own numbers do not add up (notes, "Unclear in the PDF"; ch21-facts E1): 9 operations per point on p.
          519 against 12 on p. 521, 11 additions and 6 multiplications where the listing has 12 and 8, and z left out of the
          eliminated loads.
        </p>
      </Aside>
      <p>
        What coarsening cannot remove is what each point needs for itself: one division and one square root per point per
        atom. Those are the expensive operations left in the loop.
      </p>
      <Today checked="2026-11-20 (notes/today/ch21-facts.md A10, B2, C3)">
        <p>
          CUDA's <code>/</code> and <code>sqrtf</code> are correctly rounded by default. HIP reports measured differences
          within 1 ULP of the C++ library. Metal's fast math, the compiler default per the spec, gives x / y within 2.5 ulp
          and computes sqrt as x·rsqrt(x). Not in the book: <code>charge * rsqrtf(r2)</code> replaces both with one 2-ULP
          function; compare against a reference with a tolerance.
        </p>
      </Today>
      <p>
        The cost is registers. A thread now holds four accumulators, four distances, the shared term and the charge, and
        more registers per thread can mean fewer warps resident. The book says the coarsened kernel stays within the limit,
        so occupancy does not suffer, but gives no count, compiler or device to check that against.
      </p>
      <Source section="21.3" pages="521" />
      <Today checked="2026-11-20 (notes/today/ch21-facts.md A7; ch04-facts A13, B10; ch06-facts C5)">
        <p>
          NVIDIA allows at most 255 registers per thread and 64 K per SM. See your kernel's count with{' '}
          <code>nvcc --resource-usage</code> or <code>hipcc --resource-usage</code>. Metal exposes no register count;
          Xcode shows occupancy only.
        </p>
      </Today>
      <Ask q={questionsS212['ch21/21.2/q-ex2']} label="Book exercise" />
      <Ask q={questionsS212['ch21/21.2/q-ex3']} label="Book exercise" />

      <h2>Coalescing: who owns which four</h2>
      <p>
        A profile of the coarsened kernel shows its writes are inefficient. Take one of its four write statements. Lane t
        writes point 4t + m of the block's stretch, so neighbouring lanes are four floats apart and a warp's 32 writes are
        spread over 128 floats with three skipped between each pair.
      </p>
      <Source section="21.4" pages="521" />
      <Predict
        id="ch21/21.2/p-adjacent-sectors"
        prompt={
          <>
            With the corrected adjacent ownership, one write statement of a 32-lane NVIDIA warp writes element base + 4t + m
            in lane t. How many 32-byte sectors does that one statement touch?
          </>
        }
        unit="sectors"
      >
        {(v) => (
          <p>
            <b>16.</b> The 32 lanes span 128 floats, 512 bytes, and every 32-byte sector in that span holds two of the
            floats being written. The four statements together touch 64 sectors to write 512 bytes. With the stride of Fig.
            21.10 it is 4 sectors per statement and 16 in all.{' '}
            {v === 16 ? 'That is what you had.' : v === 4 ? 'Four is the coalesced case: 32 floats side by side. Here they are four apart.' : ''}
          </p>
        )}
      </Predict>
      <p>
        The fix is to hand the points out differently. Give the block's first blockDim.x points of the stretch to its threads
        in order, then the next blockDim.x points to the same threads, and so on until each thread has four (Fig. 21.9).
        Thread t then owns t, t + blockDim.x, t + 2·blockDim.x and t + 3·blockDim.x, and in every write statement neighbouring
        lanes write neighbouring floats. Line 05 is unchanged; what changes is the write offsets and, with them, the x
        distances: <code>dx1 = dx0 + blockDim.x * gridspacing</code>. The two must change together, or the thread computes
        the right sums for the wrong points.
      </p>
      <Source section="21.4" pages="521–522" figs="Figs. 21.9, 21.10" />
      <Code
        tabs={[{ label: 'blockDim.x stride (Fig. 21.10)', code: STRIDED, tiers: TIERS, status: STATUS }]}
        caption={
          <>
            Fig. 21.10 from <code>labs/ch21/snippets/coarsen_strided.cu</code>, with a bound check per point. Compare the
            write offsets and the dx lines with the corrected Fig. 21.8 above.
          </>
        }
      />
      <p>
        Step through the write statements of one warp below under each ownership. Then change the group width and the block
        shape, and the memory unit.
      </p>
      <Simplification>
        <p>
          One warp of block (0, 0), one statement at a time, counted in the vendor's documented unit; a cache that kept a
          segment from one statement to the next is not modelled. Rows are 1,024 floats and start on a 256-byte boundary.
          Apple documents no unit, so for Metal the size is your choice. The <code>+=</code> also reads with the same pattern.
        </p>
      </Simplification>
      <PairsExplorer
        title="Write statements, one warp at a time"
        task="Step through the statements for adjacent ownership, then for the stride, then for one float4 store."
        views={['writes']}
        foot={<>Model: src/sim/dcs.ts writeStatements, counted with warpAccess from src/sim/coalesce.ts (chapter 6). Dark: bytes the lanes write in this statement; pale: earlier statements; outlined: the segments this statement touches.</>}
      />
      <p>
        Fig. 21.9 says threads compute "up to 8 potentials, skipping by half-warps". That matches neither the code (a factor
        of 4, a stride of blockDim.x) nor how coalescing is documented today: the half-warp was the coalescing unit of the
        first CUDA devices, in 2007. Do not learn a half-warp rule from it. Its other label, 64 to 256 threads per block,
        does match current NVIDIA and AMD advice.
      </p>
      <Source section="21.4" pages="522" figs="Fig. 21.9" />
      <Aside>
        <p>
          ch21-facts A3: NVIDIA's 2007 guide stated coalescing per half-warp; today's guides state it per warp, in 32-byte
          sectors. A4, B3: NVIDIA says at least 64 threads, 128 to 256 to start; AMD says multiples of 64 or 32, 128 to 256.
        </p>
      </Aside>
      <Ask q={questionsS212['ch21/21.2/q-wavefront']} />
      <p>
        How each vendor counts the cost is different, and only one of them documents it.
      </p>
      <Today checked="2026-11-20 (notes/today/ch21-facts.md B3, C4; ch06-facts A2, B1, C1)">
        <p>
          NVIDIA counts 32-byte sectors per warp; Nsight Compute's Sectors/Req is the counter, 4 being optimal for 32-bit
          accesses. AMD groups a 64- or 32-lane wavefront's addresses into 64- or 128-byte lines and advises padding rows to
          a multiple of the warp size. Apple documents no rule, so on a Mac lab 21.1 measures it.
        </p>
      </Today>
      <p>
        On Metal there is one more trap, in the stride itself. The kernel's blockDim.x becomes the threadgroup's width, and
        Metal has two ways to ask for it.
      </p>
      <Today checked="2026-11-20 (notes/today/ch05-facts.md C4, C5)">
        <p>
          With <code>dispatchThreads</code>, an edge threadgroup is smaller and <code>[[threads_per_threadgroup]]</code>{' '}
          reports its actual size. A strided mapping must take its stride from{' '}
          <code>[[dispatch_threads_per_threadgroup]]</code>, or launch with <code>dispatchThreadgroups</code> and check
          bounds.
        </p>
      </Today>
      <Ask q={questionsS212['ch21/21.2/q-metal-edge']} />
      <p>
        The book mentions a second way out for a factor of four: keep the adjacent ownership and write the four results with
        one vector store. Each thread loads its four floats as one float4, adds its four sums and stores the float4 back. A
        warp's store is then 512 contiguous bytes in one request. Try "one float4 store" in the stepper. Chapter 15 met the
        same problem for the output tiles of a matrix multiplication (<a href={href('ch15', '15.5')}>section 15.5</a>).
      </p>
      <Source section="21.4" pages="521" />
      <Today checked="2026-11-20 (notes/today/ch21-facts.md A9, C4; infra-facts A6, C7)">
        <p>
          The <code>float*</code> to <code>float4*</code> cast is NVIDIA blog practice, not in its guides, and needs
          16-byte-aligned addresses. Metal's documented way is to declare the buffer <code>device float4*</code>.
        </p>
      </Today>
      <Ask q={questionsS212['ch21/21.2/q-ex4']} label="Book exercise" />

      <h2>Measure it</h2>
      <p>
        None of this tells you how much faster the gather kernel is than the scatter kernel on your GPU, or what the stride
        is worth: no vendor documents a ratio, and Apple documents no coalescing rule at all. In{' '}
        <a href={href('ch21', 'labs')}>lab 21.1</a> you write the scatter, gather and strided kernels for CUDA, HIP and
        Metal, test them against a CPU sum, and time them against the adjacent baseline.
      </p>
    </>
  );
}
