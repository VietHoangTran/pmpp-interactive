import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Beyond, Source } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { benchRows, fmt, labStatus, num, realBackends, useReports, type BackendRun } from '../../lib/reports';
import { href } from '../../lib/router';
import { cubeAI, stencilIdealAI } from '../../sim/stencil3d';
import { sharedBytes, slabAI } from '../../sim/stencilslab';

// A kernel of our own, for one question. Checked copy: labs/ch08/snippets/regtile_two_planes.cu
// (Wrap.snippets.test.ts ties the two and pins the numbers the other questions take from the models).
const TWO_PLANES = `// Ours, not the book's: Fig. 8.12 (corrected) with its one shared plane
// doubled. Plane i is read from inCurr_s[b] and plane i + 1 is written
// to inCurr_s[1 - b], so the loop has one barrier where Fig. 8.12 has two.
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
__constant__ float c0, c1, c2, c3, c4, c5, c6;

__global__ void stencil_two_planes(const float* in, float* out,
                                   unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    int ty = threadIdx.y, tx = threadIdx.x;
    __shared__ float inCurr_s[2][IN_TILE_DIM][IN_TILE_DIM];
    float inPrev = 0.0f, inCurr = 0.0f, inNext = 0.0f;
    bool jk = j >= 0 && j < N && k >= 0 && k < N;
    if (jk && iStart >= 1 && iStart - 1 < N)
        inPrev = in[(iStart - 1)*N*N + j*N + k];
    if (jk && iStart < N)
        inCurr = in[iStart*N*N + j*N + k];
    inCurr_s[0][ty][tx] = inCurr;
    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        int b = (i - iStart) % 2;   // the buffer that holds plane i
        if (jk && i + 1 < N)
            inNext = in[(i + 1)*N*N + j*N + k];
        __syncthreads();            // the only barrier
        bool inGrid = i >= 1 && i < N - 1 && j >= 1 && j < N - 1
                   && k >= 1 && k < N - 1;
        bool inner = ty >= 1 && ty < IN_TILE_DIM - 1
                  && tx >= 1 && tx < IN_TILE_DIM - 1;
        if (inGrid && inner)
            out[i*N*N + j*N + k] = c0*inCurr
              + c1*inCurr_s[b][ty][tx - 1] + c2*inCurr_s[b][ty][tx + 1]
              + c3*inCurr_s[b][ty - 1][tx] + c4*inCurr_s[b][ty + 1][tx]
              + c5*inPrev + c6*inNext;
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[1 - b][ty][tx] = inNext;
    }
}`;

// ---------------------------------------------------------------- the numbers the questions take from the models
/**
 * Share of the ideal FLOP/B that one interior 2D tile of side t reaches (order 1): the
 * ideal is one load and one store per output, the tile loads t² and stores (t − 2)².
 * The FLOP per output cancel, so this holds for any stencil of order 1 in 2D.
 */
export const share2D = (t: number) => (2 * (t - 2) ** 2) / (t ** 2 + (t - 2) ** 2);
export const TILE_2D = 32;
/** Per cent, one decimal: 93.6 for the 32 × 32 tile. */
export const SHARE_2D_PCT = Math.round(1000 * share2D(TILE_2D)) / 10;
/** The 3D figures it is compared with: the 8 × 8 × 8 cube and the 32-wide slab as coded, as % of 13/8. */
export const SHARE_CUBE_PCT = Math.round((100 * cubeAI(8)) / stencilIdealAI(7));
export const SHARE_SLAB_PCT = Math.round((100 * slabAI(32, 30)) / stencilIdealAI(7));
/** Shared memory per 32 × 32 block: one plane (Fig. 8.12), two (the kernel above), three (Fig. 8.10). */
export const ONE_PLANE = sharedBytes('registers', 32);
export const TWO_PLANES_BYTES = 2 * ONE_PLANE;
export const THREE_PLANES = sharedBytes('planes', 32);

const n = (v: number) => v.toLocaleString('en-US');

export const questionsWrap: Record<string, Question> = {
  'ch08/wrap/q-2d-tile': {
    kind: 'numeric',
    id: 'ch08/wrap/q-2d-tile',
    prompt: (
      <>
        A 2D 5-point stencil is tiled the way Fig. 8.8 tiles the 3D one: one thread per input element, {TILE_2D} ×{' '}
        {TILE_2D} input tiles, {TILE_2D - 2} × {TILE_2D - 2} outputs per block. What share of the 2D stencil's ideal
        FLOP/B does one interior block reach? Answer in per cent.
      </>
    ),
    answer: SHARE_2D_PCT,
    tol: 0.01,
    unit: '%',
    explain: (
      <p>
        Per block: {n((TILE_2D - 2) ** 2)} outputs, {n(TILE_2D ** 2)} loads and {n((TILE_2D - 2) ** 2)} stores. The ideal
        is one load and one store per output, so the block reaches 2 × {n((TILE_2D - 2) ** 2)} / ({n(TILE_2D ** 2)} +{' '}
        {n((TILE_2D - 2) ** 2)}) = {SHARE_2D_PCT}% of it; the operations per output cancel. The 8 × 8 × 8 cube of page
        8.4 reached {SHARE_CUBE_PCT}% of its ideal, and the 32-wide slab of page 8.5, as coded, {SHARE_SLAB_PCT}%. In two
        dimensions the thread limit still allows a wide tile, so tiling alone gets close. Coarsening is the answer to the
        third dimension.
      </p>
    ),
    source: 'method of ch. 8 §8.4, p. 192 (the 2D ideal, 1.125 FLOP/B, is on p. 190); the 2D case is ours',
  },
  'ch08/wrap/q-box27': {
    kind: 'choice',
    id: 'ch08/wrap/q-box27',
    prompt: (
      <>
        A 27-point stencil reads every point of the 3 × 3 × 3 box around the centre, edges and corners included. You
        want both of Fig. 8.12's moves for it: march a 2D block through z, and keep the planes before and after the
        current one in registers. What still works?
      </>
    ),
    options: [
      {
        label: 'Both moves, unchanged: the window is still three planes deep.',
        why: 'In Fig. 8.12 a thread reads only its own point of the planes before and after. Here it reads nine points of each, eight of them loaded by other threads.',
      },
      {
        label: 'Marching works, but the planes before and after must stay in shared memory: up to nine threads read each of their values.',
        correct: true,
      },
      {
        label: 'Neither: a box stencil reads corners, so it needs the cube tile of Fig. 8.8.',
        why: 'Marching does not depend on the stencil\'s shape. The block still holds a rolling window of three planes; it just reads more of each.',
      },
      {
        label: 'Register tiling works, but marching does not: the corners lie in planes the window does not hold.',
        why: 'The stencil still has order 1 in z: planes z − 1, z and z + 1 hold every value an output needs, corners included.',
      },
    ],
    explain: (
      <p>
        The book limits register tiling to stencils whose neighbours lie on the axes through the centre (p. 196), and
        the reason is who reads a value. On the axes, only the thread at the same (x, y) reads a value of the planes
        before and after, so it can live in that thread's registers. With edges and corners in the stencil, the thread
        at (x, y) also reads (x ± 1, y ± 1) of those planes, which other threads loaded, so all three planes must be
        shared. Coarsening still pays as before: the block loads exactly the same planes, and only reads more of them.
      </p>
    ),
    source: 'ch. 8 §8.6, p. 196 (axis-only stencils); the 27-point case is ours',
  },
  'ch08/wrap/q-two-planes': {
    kind: 'choice',
    id: 'ch08/wrap/q-two-planes',
    prompt: (
      <>
        The kernel above is Fig. 8.12 with its shared plane doubled: plane i is read from <code>inCurr_s[b]</code>, and
        plane i + 1 is written into <code>inCurr_s[1 - b]</code>. It has one barrier per iteration where Fig. 8.12 has
        two. Is it correct under every thread order, and what did the change cost?
      </>
    ),
    options: [
      {
        label: 'Wrong: a fast thread can overwrite the plane its neighbours are still reading.',
        why: 'It writes the other buffer. The buffer being read is written again only at the end of the next iteration, after a barrier that no thread passes until every thread has finished this iteration\'s reads.',
      },
      {
        label: 'Correct, and free: Fig. 8.12\'s second barrier was never needed.',
        why: 'In Fig. 8.12 the write after the compute goes into the same inCurr_s the neighbours are reading. Without that barrier a fast thread overwrites a value a slow neighbour has not read yet.',
      },
      {
        label: `Correct, at the price of a second plane of shared memory: ${n(TWO_PLANES_BYTES)} bytes per 32 × 32 block instead of ${n(ONE_PLANE)}.`,
        correct: true,
      },
      {
        label: 'Wrong: with one barrier a thread can read plane i + 1 before its neighbours have written it.',
        why: 'That is what the remaining barrier prevents. It sits between the writes of plane i + 1, at the end of one iteration, and the reads of it in the next.',
      },
    ],
    explain: (
      <p>
        Fig. 8.12's second barrier protects a false dependence (write after read): the next plane is written into the
        array the neighbours are still reading. <a href={href('ch06', '6.7')}>Chapter 6's double buffering</a> removes it by writing somewhere else; the
        barrier that remains protects the true dependence. The price is one more {n(ONE_PLANE)}-byte plane, which still
        leaves the block below Fig. 8.10's three planes ({n(THREE_PLANES)} bytes). Whether one barrier fewer pays for
        the extra plane is a measurement, not something the model says.
      </p>
    ),
    source: 'ch. 6 §6.7, pp. 146–147 (double buffering); ch. 8 §8.6, p. 197; the kernel is ours',
  },
  'ch08/wrap/q-decode-placement': {
    kind: 'choice',
    id: 'ch08/wrap/q-decode-placement',
    prompt: (
      <>
        A decode-step attention kernel gives one block to one head. The head's cached keys are 128 floats each, stored
        one after another. Each thread scores some of the keys against the head's query vector, also 128 floats, and
        keeps a running maximum and a running sum over its own keys. By this chapter's rule, where does each value
        belong?
      </>
    ),
    options: [
      {
        label: 'All of it in shared memory, the query, the keys and the running values, since shared memory is faster than global.',
        why: 'Shared memory is for values several threads read, or for staging data whose global loads would not coalesce. A running maximum is read by one thread only and changes at every key: keeping it in shared memory adds traffic and nothing else.',
      },
      {
        label: "The query in shared memory, each thread's running maximum and sum in its registers, and no key shared.",
        correct: true,
      },
      {
        label: 'The query copied into every thread\'s registers, 128 floats each, so that no thread waits on shared memory.',
        why: 'Every thread would hold the same 128 values: the register budget of page 8.5 spent on copies. At 1,024 threads, CUDA allows at most 64 registers per thread.',
      },
      {
        label: 'The keys in shared memory, so that the block can reuse them across threads, and the running values in registers.',
        why: 'No key is reused here: each is scored once, against one query. Reuse appears when one block scores several queries against the same keys, for instance several query heads that share one set of keys; then sharing them pays. (Staging keys through shared memory can still pay for another reason; see the explanation.)',
      },
    ],
    explain: (
      <>
        <p>
          Ask who reads each value. Every thread reads the query: shared memory, loaded once per block. Only its own
          thread reads a running maximum or sum: registers, like <code>inPrev</code> and <code>inNext</code>. One thread
          reads each key, once: by this chapter's rule there is nothing to share.
        </p>
        <p>
          The rule settles sharing. It does not settle how a key gets from global memory to its thread, and this layout
          gets that wrong: with one thread per key, neighbouring lanes read addresses 128 floats (512 bytes) apart, an
          uncoalesced load in chapter 6's terms. Chapter 6 gives two ways out. Change the mapping, so that a warp's lanes
          split each key across the head dimension and neighbouring lanes read neighbouring floats (they then add up
          their partial dot products). Or load the keys into shared memory with coalesced loads and do the strided reads
          there, as corner turning does (§6.1), even though each key has a single reader. Applying them to attention is
          our sketch, not a survey of real kernels; chapter 20 builds attention kernels for milestone M6.
        </p>
      </>
    ),
    source:
      'the rule of ch. 8 §8.6, p. 196; coalescing strategies in ch. 6 §6.1, p. 129; the attention setting and the layouts are ours; register budget in notes/today/ch08-facts.md A4',
  },
  'ch08/wrap/q-cache-basic': {
    kind: 'choice',
    id: 'ch08/wrap/q-cache-basic',
    prompt: (
      <>
        Lab 8.1 reports each kernel's speed as "GB/s at 8 B/point": interior points per second times 8 bytes, as if every
        point cost one load and one store. Suppose the basic kernel's figure comes out close to the bandwidth that lab
        5.1's probe measured on the same device. What follows?
      </>
    ),
    options: [
      {
        label: 'The basic kernel requests only 8 bytes per point, so the model\'s count of 32 was wrong.',
        why: 'It requests 32: seven loads and one store. The code fixes what is requested; the hardware decides how much of it reaches DRAM.',
      },
      {
        label: 'Most of its 32 requested bytes per point never reach DRAM, so register tiling has little DRAM traffic left to save.',
        correct: true,
      },
      {
        label: 'The device has bandwidth to spare, so the register-tiled kernel will run about 3.6 times faster, as the byte count predicts.',
        why: 'The kernel already moves data at about the rate the probe found. 3.6 times faster would mean about 2 bytes of DRAM traffic per point, less than one load and one store.',
      },
      {
        label: 'The measurement must be wrong, since 0.41 FLOP/B caps the kernel at a quarter of that.',
        why: 'That cap assumes every requested byte comes from DRAM. Caches break the assumption, as the book notes (p. 189).',
      },
    ],
    explain: (
      <p>
        Every correct sweep of a grid much larger than the caches reads each input from DRAM and writes each output to
        it at least once: about 8 bytes per point. If the basic kernel's points per second times 8 bytes is already near
        the bandwidth the device delivers, it cannot be moving much more than that, whatever it requests. Register
        tiling cuts requests, not that floor. On such a device any gain has to come from fewer cache accesses.
      </p>
    ),
    source: 'ch. 8 §8.3, p. 189 (caches raise the measured ratio); the reasoning is ours; measurements from labs 5.1 and 8.1',
  },
};

// ---------------------------------------------------------------- this machine's numbers for the last question
/** The rows of one lab on one backend: the reader's own where they passed, else the reference's. */
function preferredRows(run: BackendRun | undefined, lab: string) {
  const mine = labStatus(run, lab, 'work') === 'pass' ? benchRows(run, lab, 'work') : [];
  return mine.length ? mine : benchRows(run, lab, 'solution');
}

function OnThisMachine() {
  const { reports, loading } = useReports();
  const lines = realBackends(reports).flatMap((b) => {
    const basic = preferredRows(reports[b], 'ch08/lab1_stencil3d').find((r) => r.name === 'basic');
    const roof = preferredRows(reports[b], 'ch05/lab1_roofline').find((r) => r.name === 'roofline');
    const a = num(basic?.gbps_compulsory);
    const bw = num(roof?.bandwidth_gb_per_s);
    if (!a || !bw) return [];
    return [{ b, device: String(reports[b].device.name ?? b), a, bw, n: basic?.n }];
  });
  if (loading) return null;
  return lines.length ? (
    <>
      {lines.map((l) => (
        <p key={l.b} data-machine={l.b} data-ratio={fmt(l.a / l.bw)}>
          Your own numbers, {l.device} ({l.b}): the basic kernel of lab 8.1 ran at <b>{fmt(l.a)} GB/s at 8 B/point</b> on a
          grid of side {String(l.n)}, and lab 5.1's bandwidth probe measured <b>{fmt(l.bw)} GB/s</b>: a ratio of{' '}
          {fmt(l.a / l.bw)}. The labs page has the register-tiled kernel's figure beside it.
        </p>
      ))}
    </>
  ) : (
    <p className="muted" data-machine="none">
      To put your own device into this question you need benchmark rows from lab 8.1 and from lab 5.1 on the same
      backend: <code>python3 labs/pmpp.py run 8.1 --bench</code> and <code>python3 labs/pmpp.py run 5.1 --bench</code>.
    </p>
  );
}

// ---------------------------------------------------------------- the page
export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = realBackends(reports);
  const lab = 'ch08/lab1_stencil3d';
  const passing = [...backends, 'emu'].filter((b) => labStatus(reports[b], lab, 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch08/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul>
        <li>
          A tile's ratio is set by its halo. A small cube is mostly halo (58% at 8 × 8 × 8), and with one thread per
          element the thread limit, not the formula, decides how large the cube can get.
        </li>
        <li>
          Coarsen along one axis to escape that limit: a block that is one plane of the tile marches through the third
          dimension, holding a rolling window of planes on chip. The tile grows without more threads, and the slab depth
          becomes a knob that trades ratio for the number of blocks.
        </li>
        <li>
          A shared buffer reused across iterations needs two barriers per iteration: one before the reads (read after
          write), one before the overwrite (write after read). Every thread must reach both, so guard loads and stores and
          never return early. A second buffer removes the second barrier.
        </li>
        <li>
          A value that exactly one thread reads belongs in that thread's registers; only values that several threads read
          need shared memory. That saves shared memory and on-chip traffic, costs registers, and leaves global traffic
          unchanged.
        </li>
        <li>
          Keep a per-thread window at constant indices, as named scalars or a fully unrolled loop. NVIDIA documents a
          run-time-indexed array as a candidate for local memory, which is device memory.
        </li>
        <li>
          Registers are a block budget. At 1024 threads CUDA allows at most 64 per thread and fails the launch above that
          without <code>__launch_bounds__</code>; HIP limits registers and may spill; Metal may lower the pipeline's
          thread cap. Read the limits;
          do not assume them.
        </li>
        <li>
          The model counts requested bytes; caches decide what reaches DRAM. Predict from the model, then measure.
        </li>
      </ul>
      <Source section="8.7" pages="198" />

      <h2>A mixed check</h2>
      <p>
        These draw on the whole chapter and on situations the pages did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
      </p>
      <Beyond>
        <p>
          The 2D and 27-point cases, the two-plane kernel and the attention setting are ours. The two-plane kernel is
          front-end-checked and has not been run; a unit test runs a model of its loop one thread at a time under three
          orders.
        </p>
      </Beyond>
      <Ask q={q['ch08/wrap/q-2d-tile']} label="Work it out" />
      <Ask q={q['ch08/wrap/q-box27']} />
      <Code
        tabs={[
          {
            label: 'CUDA (not a lab file: for this question only)',
            code: TWO_PLANES,
            status: 'front-end-checked; never run',
            tiers: {
              in: 'global', out: 'global', inCurr_s: 'shared',
              inPrev: 'reg', inCurr: 'reg', inNext: 'reg', i: 'reg', j: 'reg', k: 'reg', iStart: 'reg', b: 'reg', ty: 'reg', tx: 'reg', jk: 'reg',
              c0: 'const', c1: 'const', c2: 'const', c3: 'const', c4: 'const', c5: 'const', c6: 'const',
            },
          },
        ]}
      />
      <Ask q={q['ch08/wrap/q-two-planes']} label="Read the code" />
      <Ask q={q['ch08/wrap/q-decode-placement']} />
      <Ask q={q['ch08/wrap/q-cache-basic']} />
      <OnThisMachine />

      <h2>The book's exercises</h2>
      <p>
        The chapter ends with two exercises in nine parts (pp. 198–199), both about sizing the chapter's kernels: the
        first for a 120 × 120 × 120 grid, the second for one coarsened block. Every part is worked on page 8.4 or 8.5;
        the book prints no answers, so the answers there are ours.
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
              <td>1a</td>
              <td>Output points computed per sweep</td>
              <td>a question in <a href={href('ch08', '8.4')}>8.4</a></td>
            </tr>
            <tr>
              <td>1b</td>
              <td>Blocks for the basic kernel with 8 × 8 × 8 blocks</td>
              <td>
                a question in <a href={href('ch08', '8.4')}>8.4</a>; the basic launch of{' '}
                <a href={href('ch08', 'labs')}>lab 8.1</a>
              </td>
            </tr>
            <tr>
              <td>1c</td>
              <td>Blocks for the cube-tiled kernel with 8 × 8 × 8 blocks</td>
              <td>a question in <a href={href('ch08', '8.4')}>8.4</a></td>
            </tr>
            <tr>
              <td>1d</td>
              <td>Blocks for the z-coarsened kernel with 32 × 32 blocks</td>
              <td>a question in <a href={href('ch08', '8.5')}>8.5</a></td>
            </tr>
            <tr>
              <td>2a, 2b</td>
              <td>Input and output elements of a block coarsened by 16</td>
              <td>
                a question each in <a href={href('ch08', '8.5')}>8.5</a>, and the slab explorer's "Loads and stores per
                block" at D = 16
              </td>
            </tr>
            <tr>
              <td>2c</td>
              <td>That block's FLOP/B</td>
              <td>a question in <a href={href('ch08', '8.5')}>8.5</a>, and the slab explorer at D = 16</td>
            </tr>
            <tr>
              <td>2d, 2e</td>
              <td>Shared memory per block without and with register tiling</td>
              <td>
                a question in <a href={href('ch08', '8.5')}>8.5</a>, and the explorer's shared-memory readout for both
                kernels
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Where the PDF looks wrong</h2>
      <p>
        The copy of the book this course was built from has a few apparent slips in chapter 8, and a few statements that
        current documentation does not support. They are listed so that you are not thrown by them when you read the
        book alongside.
      </p>
      <ul data-list="errata">
        <li>
          Figs. 8.10 and 8.12 pair <code>c3</code> with the y + 1 neighbour and <code>c4</code> with y − 1, the reverse
          of Figs. 8.6 and 8.8; the results differ unless the two weights are equal. The course and lab 8.1 follow Figs.
          8.6 and 8.8 (<a href={href('ch08', '8.4')}>8.4</a>, <a href={href('ch08', '8.5')}>8.5</a>).
        </li>
        <li>
          Fig. 8.12's last line stores <code>inNext_s</code>, which that kernel does not declare; the register{' '}
          <code>inNext</code> is meant. Its line numbers near the end run 33, 34, 35, 34, 35, 35, and the text's "line
          34" is the second 34 (<a href={href('ch08', '8.5')}>8.5</a>).
        </li>
        <li>
          p. 195 gives the coarsened kernel 1.52 FLOP/B at t = 32. The book's own formula and Fig. 8.10 as coded give
          1.47; 1.52 is the limit for a slab of unbounded depth (<a href={href('ch08', '8.5')}>8.5</a>).
        </li>
        <li>
          p. 196 says register tiling cuts shared memory to a third of "that by the kernel in Fig. 8.12"; Fig. 8.10 is
          meant.
        </li>
        <li>
          <code>IN_TILE_DIM</code> and <code>OUT_TILE_DIM</code> are used in Figs. 8.8, 8.10 and 8.12 and defined in
          none; the text gives output = input − 2 (p. 192). Lab 8.1 defines both. No listing shows host code.
        </li>
        <li>
          p. 186 says the stencils in "Fig. 8.3(a), (b), and (c)" have orders 1, 2 and 3; Fig. 8.2 is meant (Fig. 8.3's
          caption gives 1, 2, 1, 2).
        </li>
        <li>
          p. 187 describes the 2D 5-point stencil as using two grid points on each side of the centre along each axis.
          It uses one on each side, two per axis.
        </li>
        <li>
          p. 184 calls 0.87 "the x value that corresponds to element F[2]"; 0.87 is the stored value, sin(2π/6), and the
          x value is 2π/6.
        </li>
        <li>
          Small slips: "compared 6.25 FLOP/B" (p. 190), "a copy the current plane" (p. 196), and{' '}
          <code>N - 1&amp;&amp; k</code> without a space in Fig. 8.6.
        </li>
        <li>
          Not an error, but unexplained: Fig. 8.5 leaves the input grid's four corner cells unshaded. A 5-point stencil
          never reads them, the "no corners" point of <a href={href('ch08', '8.4')}>8.4</a>.
        </li>
      </ul>
      <p>And where current vendor documentation says something else:</p>
      <ul data-list="dated">
        <li>
          p. 192, "the practical limit on t is 8": a 10 × 10 × 10 block of 1000 threads is within every documented CUDA
          and HIP limit. 8 is the book's power-of-two choice (<a href={href('ch08', '8.4')}>8.4</a>).
        </li>
        <li>
          p. 193, the 8 × 8 × 8 tile's loads are "not coalesced": by NVIDIA's 32-byte model they are half efficient, and
          the suggested 32 × 8 × 4 block reaches 80%, not 100%, because of the halo's − 1 (
          <a href={href('ch08', '8.4')}>8.4</a>).
        </li>
        <li>
          pp. 184–185, single and half precision have "much higher" throughput than double: NVIDIA's FP32 to FP64 ratio
          runs from 2:1 to 64:1 depending on the compute capability, and Metal has no <code>double</code>.
        </li>
        <li>
          p. 197, registers have "significantly lower latency and higher bandwidth" than shared memory: NVIDIA documents
          the latency order and no figures; Apple documents nothing about registers.
        </li>
        <li>
          Missing from the book: on CUDA a 32 × 32 block whose kernel needs more than 64 registers per thread fails to
          launch unless the kernel carries <code>__launch_bounds__</code> (<a href={href('ch08', '8.5')}>8.5</a>).
        </li>
      </ul>
      <Aside>
        <p>
          These are our readings of one PDF, checked against current vendor documentation on 2026-11-20
          (notes/today/ch08-facts.md). If your copy differs, trust your copy.
        </p>
      </Aside>

      <h2>What this chapter left out</h2>
      <p>
        The chapter is condensed in this course, so here is what is not covered and where to find it. Most of §8.1's
        background (pp. 184–187): sampling sin x on a grid (Fig. 8.1), structured and unstructured grids, interpolation,
        the finite-difference derivative and its error, and the stencil drawings of Figs. 8.2 to 8.4; the start page
        keeps two sentences of it. Most of the opening's list of differences from convolution (p. 183): that stencils
        serve iterative solvers, that their weights come from the differential equation, and the ordering constraints
        some solvers impose between outputs, of which the start page keeps only the assumption that rules them out
        (§8.2). Two points of that list do survive: stencil patterns that allow optimisations convolution does not
        (register tiling, page 8.5), and precision, which with §8.1's discussion of it (pp. 184–185) becomes the
        bytes-per-value setting in the explorers. The 3D-grid point is the §8.7 summary's (p. 198), and it is the
        reason for page 8.5.
      </p>
      <p>
        Figs. 8.9 and 8.11 are redrawn by page 8.5's animation and stepper rather than reproduced. You do not write the
        cube-tiled kernel of Fig. 8.8 or the three-plane kernel of Fig. 8.10: lab 8.1 provides both, corrected and tested.
        Higher-order 3D stencils appear only in the explorers and in a few questions, and the 32 × 8 × 4 block only in
        page 8.4's warp explorer; no kernel is built for either. Nothing later in the course depends on what was left
        out.
      </p>

      <h2>The project so far</h2>
      <p>
        This chapter adds no milestone to the through-line project. Its two techniques are on the way to one:{' '}
        <a href={href('ch15', 'start')}>chapter 15</a>'s coarsened, register-tiled matrix multiplication is milestone M3,
        built on the same rule about who reads a value.
        Lab 8.1 is optional.{' '}
        <span data-lab81={passing.join(',') || 'none'}>
          {backends.length === 0 && !reports['emu']
            ? 'No lab results are recorded on this machine yet.'
            : passing.length
              ? `Your own version passes on ${passing.map(where).join(' and ')} on this machine.`
              : 'Your own version has not passed on this machine yet.'}
        </span>{' '}
        See the <a href={href('ch08', 'labs')}>lab</a>.
      </p>
      <p>
        Chapter 9 leaves a habit of every kernel so far behind: each output belonged to exactly one thread. In a
        histogram, which bin a thread updates depends on the data it reads, so threads collide on outputs. That needs
        atomic operations and, to make them fast, privatisation: per-block copies of the output in shared memory. Its
        kernel is also the first piece of the sampling path, the token counts behind a repetition penalty (milestone M5).
        Start at <a href={href('ch09', 'start')}>chapter 9</a>.
      </p>
    </>
  );
}
