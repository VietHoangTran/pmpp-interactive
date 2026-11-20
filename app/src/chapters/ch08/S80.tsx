import { Ask, Predict, type Question } from '../../components/Ask';
import { Beyond, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { haloCount, haloFraction, haloFraction2D, threadsCube } from '../../sim/stencil3d';

// The start page of chapter 8: what the chapter adds, the two assumptions every kernel in
// it relies on, a map, and a check on the four ideas from chapters 4 to 7 it builds on.
// Every number below comes from src/sim/stencil3d.ts.

const T3 = 8;
const HALO3 = haloCount(T3); // 296
const PCT3 = 100 * haloFraction(T3); // 57.8
const PCT2 = 100 * haloFraction2D(32); // 12.1
const pct = (v: number) => (Math.round(v * 10) / 10).toString();

export const questionsS80: Record<string, Question> = {
  'ch08/start/q-warp-rows': {
    kind: 'choice',
    id: 'ch08/start/q-warp-rows',
    prompt: (
      <>
        A block has 8 × 8 × 8 threads. On an NVIDIA GPU, which threads make up its first warp of 32?
      </>
    ),
    options: [
      {
        label: (
          <>
            <code>threadIdx.z = 0</code>, <code>threadIdx.y</code> from 0 to 3, <code>threadIdx.x</code> from 0 to 7:
            four rows of eight.
          </>
        ),
        correct: true,
      },
      {
        label: (
          <>
            <code>threadIdx.x = 0</code>, <code>threadIdx.y</code> from 0 to 3, <code>threadIdx.z</code> from 0 to 7: a
            column through the block.
          </>
        ),
        why: 'That would make z the fastest index. The order is the other way round.',
      },
      {
        label: (
          <>
            All 64 threads with <code>threadIdx.z = 0</code>: one full plane.
          </>
        ),
        why: 'A plane of 8 × 8 is 64 threads, two warps of 32. (It is one wavefront on an AMD GPU with 64-wide wavefronts, which page 8.4 comes back to.)',
      },
      {
        label: 'There is no rule: how threads are grouped is implementation-defined on every GPU.',
        why: 'NVIDIA documents the rule. Of the three backends in this course, Metal is the one that leaves the grouping into SIMD-groups undefined.',
      },
    ],
    explain: (
      <p>
        Threads are put in a line with x fastest, then y, then z, and the line is cut into runs of 32. With 8 threads per
        row, 32 consecutive threads are four rows of the same z-plane. Hold on to the picture: on page 8.4 those four rows
        are four separate stretches of global memory, and that is the coalescing problem of a cube-shaped tile.
      </p>
    ),
    source: 'ch. 4 §4.4, pp. 75–76; notes/today/ch08-facts.md (the linear order is settled in ch04 A8; Metal in ch04 C2)',
  },
  'ch08/start/q-false-dep': {
    kind: 'choice',
    id: 'ch08/start/q-false-dep',
    prompt: (
      <>
        A block runs a loop. Each iteration, every thread writes one element of a shared buffer, waits at a barrier,
        computes from elements other threads wrote, and waits at a second barrier. The next iteration overwrites the same
        buffer. If every iteration had a buffer of its own, which barrier could you drop?
      </>
    ),
    options: [
      {
        label: 'The one after the writes.',
        why: 'Threads still read what other threads wrote. They must not read it before it is there, whatever buffer it is in.',
      },
      { label: 'The one after the compute.', correct: true },
      { label: 'Both.', why: 'One of them guards a dependence that more memory does not remove.' },
      { label: 'Neither.', why: 'One of them exists only because the buffer is reused.' },
    ],
    explain: (
      <p>
        The first barrier guards a true dependence: readers need what writers produce (read after write). The second
        guards a false one: the next iteration's writers need nothing from this iteration's readers; they only must not
        overwrite what is still being read (write after read). Page 8.5 has a loop like this one, with three buffers
        whose contents shift by one plane each iteration (by copying), and two barriers in every iteration.
      </p>
    ),
    source: 'ch. 5 §5.4, pp. 110–111',
  },
  'ch08/start/q-unroll-regs': {
    kind: 'choice',
    id: 'ch08/start/q-unroll-regs',
    prompt: (
      <>
        After coarsening, each thread keeps four partial results in <code>float acc[4]</code> and updates{' '}
        <code>acc[c]</code> in a loop over <code>c</code> from 0 to 3. Why does the book want that loop fully unrolled?
      </>
    ),
    options: [
      {
        label: (
          <>
            With only constant indices (<code>acc[0]</code> to <code>acc[3]</code>) the compiler can keep the array in
            registers.
          </>
        ),
        correct: true,
      },
      { label: 'Unrolling removes a barrier from the loop.', why: 'There is no barrier in this loop; unrolling copies the body, it does not change what synchronises.' },
      { label: 'Unrolling raises the kernel\'s FLOP/B.', why: 'The same bytes move and the same operations run. Where acc lives changes, not the global traffic.' },
      { label: 'Unrolling is needed for the loads to coalesce.', why: 'Coalescing depends on which addresses the lanes of a warp touch at once, not on the loop structure inside one thread.' },
    ],
    explain: (
      <p>
        An array indexed by a value known only at run time has to live somewhere addressable. The book says such local
        arrays go to global memory by default; NVIDIA's documentation calls it local memory and says it has the latency
        and bandwidth of device memory. Indexed only by constants, the four elements can be four registers. Page 8.5
        relies on this: its register-tiled kernel keeps each thread's values from three consecutive planes in three named
        variables, not in an array.
      </p>
    ),
    source: 'ch. 6 §6.6, pp. 145–146; NVIDIA\'s rule for local memory in notes/today/ch08-facts.md A5',
  },
  'ch08/start/q-halo2d': {
    kind: 'numeric',
    id: 'ch08/start/q-halo2d',
    prompt: (
      <>
        A block loads a 32 × 32 input tile for a 3 × 3 convolution (radius 1). How many of the loaded elements are halo,
        that is, loaded but not the centre of any output the block computes?
      </>
    ),
    answer: Math.round(1024 * haloFraction2D(32)),
    unit: 'elements',
    explain: (
      <p>
        The output tile is the input tile less one layer on each side: 30 × 30 = 900. So 1024 − 900 = 124 elements are
        halo, about {Math.round(PCT2)}% of the tile. The book quotes this figure in chapter 8 to compare it with a 3D tile.
      </p>
    ),
    source: 'ch. 7 §7.5, pp. 172–176; the numbers as the book gives them in ch. 8 §8.4, p. 193',
  },
};

/** The reveal after the halo prediction, written for what the reader committed. */
function HaloReveal({ v }: { v: number }) {
  const close = Math.abs(v - PCT3) <= 4;
  const twoD = Math.abs(v - (100 - 100 * (6 / 8) ** 2)) <= 2.5; // 43.75: the 2D share at this width
  const oneD = Math.abs(v - 25) <= 2.5; // 1 − 6/8
  const carried = v <= PCT2 + 6; // the 2D figure carried over
  return (
    <>
      <p>
        <b>
          {HALO3} of the {threadsCube(T3)} loaded elements are halo: {pct(PCT3)}%.
        </b>{' '}
        The output tile is 6 × 6 × 6 = 216, and the rest of the 8 × 8 × 8 tile is there only to feed it.
      </p>
      {close ? (
        <p>That is what you predicted. More than half of what the block loads is not the centre of any output.</p>
      ) : twoD ? (
        <p>
          You have the share for an 8 × 8 square: 1 − (6/8)² = 43.75%. The third axis multiplies the interior share by
          another 6/8, so the interior is (6/8)³ = 42% of the cube and the halo the other 58%.
        </p>
      ) : oneD ? (
        <p>
          That is the share along one axis, 2 of 8. The interior is 6/8 of the tile along each of three axes, so it is
          (6/8)³ = 42% of the cube, and the halo is the rest.
        </p>
      ) : carried ? (
        <p>
          The 2D figure does not carry over. A 32 × 32 tile is wide; an 8 × 8 × 8 tile is not, because the thread limit
          that allows 32 × 32 = 1024 in two dimensions allows only about 10 per side in three. A thin tile is mostly
          surface: (6/8)³ = 42% interior.
        </p>
      ) : (
        <p>
          The interior is 6/8 of the tile along each of three axes, so it is (6/8)³ = 42% of the cube, and the halo is
          the other 58%.
        </p>
      )}
      <p>
        Against {pct(PCT2)}% for the 2D tile above, that is the problem <a href={href('ch08', '8.4')}>page 8.4</a>{' '}
        takes apart. Its explorer "A cube-shaped tile" lets you change the side and watch the share fall, and shows what
        stops you making the side larger.
      </p>
    </>
  );
}

export function S80() {
  return (
    <>
      <p>
        A transformer has no three-dimensional grids to sweep, so the computation this chapter is named after matters
        little to you. Read it for the two techniques the book develops on it, both of which transfer to any kernel you
        will write:
      </p>
      <ul>
        <li>
          <b>thread coarsening along one dimension</b>: a block of threads walks through the data one slice at a time and
          keeps only a small rolling window of slices on chip, so the tile it covers can grow without more threads;
        </li>
        <li>
          <b>register tiling decided by who reads a value</b>: a value that exactly one thread reads belongs in that
          thread's registers; only values that several threads read need shared memory.
        </li>
      </ul>
      <Source section="8 (opening), 8.1, 8.2" pages="183–188" />
      <p>
        Along the way it gives a third, smaller thing: the FLOP/B bookkeeping for a tile with a halo in three
        dimensions, where the halo turns out to be most of the tile. The book itself ties the chapter's coarsening to
        matrix multiplication, in chapter 6 (p. 153) and again in chapter 15 (p. 352). That is where you will use it: the
        coarsened, register-tiled GEMM of chapter 15 is milestone M3.
      </p>

      <h2>The background in two sentences</h2>
      <p>
        A <b>stencil</b> computes each point of a regular grid from that point and a few neighbours along the axes, with
        weights that come from approximating derivatives by finite differences. A solver applies it to every point of
        the grid, one <b>sweep</b> after another. The number of neighbours on each side of the centre is the stencil's{' '}
        <b>order</b>: the 3D stencil of this chapter has order 1, so it reads the point and its six axis neighbours, 7
        points in all.
      </p>
      <p>
        Every kernel in the chapter rests on two simplifying assumptions. No output of a sweep depends on another output
        of the same sweep, so all outputs can be computed in parallel from the input grid. And the points on the
        grid's boundary hold boundary conditions: they keep their input values, and only interior points are computed.
      </p>
      <Source section="8.2" pages="187–188" figs="Fig. 8.5" />

      <h2>Where this sits in an inference stack</h2>
      <p>
        One place in LLM inference does run a stencil. Mamba-style layers apply a short causal convolution along the
        token axis, and a convolution along one axis is a 1D stencil. In llama.cpp the CUDA kernel for it,{' '}
        <code>ssm_conv</code>, uses this chapter's moves. Its short-sequence path coarsens along the tokens: each thread
        marches through one channel with a per-thread window of the last few inputs (whether that window stays in
        registers was not checked). Its long-sequence path is the halo tile of page 8.4: a block loads a tile of tokens
        plus its halo into shared memory. Page 8.5 comes back to it in code.
      </p>
      <Beyond>
        <p>
          The link to Mamba-style layers and llama.cpp is ours, not the book's. The kernel's structure was read from the
          llama.cpp source (notes/today/ch08-facts.md, D1).
        </p>
      </Beyond>
      <p>
        The kernel is <code>ggml/src/ggml-cuda/ssm-conv.cu</code>, and the same file builds llama.cpp's HIP backend. Its
        window is indexed by the token counter, the run-time index the third question below is about.
      </p>
      <Today checked="2026-11-20 (notes/today/ch08-facts.md, D1)">
        <p>
          Read on the llama.cpp default branch; the file's last commit was on 2026-11-18. The short-sequence kernel is{' '}
          <code>ssm_conv_f32</code>, the long-sequence one <code>ssm_conv_long_token_f32</code>.
        </p>
      </Today>

      <h2>The chapter in this course</h2>
      <p>
        Two pages. <a href={href('ch08', '8.4')}>Page 8.4</a> counts what the one-thread-per-point kernel and the
        cube-tiled kernel ask of memory, and shows why a cube-shaped tile cannot get far: the thread limit caps its side,
        the halo takes more than half of it, and each warp's load is split across rows (book pp. 187–193).{' '}
        <a href={href('ch08', '8.5')}>Page 8.5</a> breaks the link between tile size and thread count by marching a
        plane of threads through z, then moves the values only one thread reads into registers (pp. 193–198). Then one
        optional <a href={href('ch08', 'labs')}>lab</a>, which builds both kernels and measures them, and the{' '}
        <a href={href('ch08', 'wrap')}>summary</a> with the book's exercises.
      </p>
      <p>
        What is condensed: the finite-difference background of §8.1 (pp. 184–187), with the sampled sine curve of Fig.
        8.1, the derivation of the 3-point stencil and the drawings of 1D, 2D and 3D stencils in Figs. 8.2 to 8.4, is
        reduced to the two paragraphs above. The book's remarks on precision (pp. 184–185) appear on page 8.4 only as
        what an 8-byte value does to the ratios. Read those pages if you want the numerical-methods side; nothing later in
        this course depends on them.
      </p>

      <h2>Before you start</h2>
      <p>
        Four questions on what this chapter takes from chapters 4 to 7. Each one is used on a later page, and each one
        comes back in <a href={href('review')}>review</a>.
      </p>
      <Ask q={questionsS80['ch08/start/q-warp-rows']} label="From chapter 4" />
      <Ask q={questionsS80['ch08/start/q-false-dep']} label="From chapter 5" />
      <Ask q={questionsS80['ch08/start/q-unroll-regs']} label="From chapter 6" />
      <Ask q={questionsS80['ch08/start/q-halo2d']} label="From chapter 7" />
      <p>
        Now the same count in three dimensions. A 3D tile for a stencil of order 1 has to be small, because one thread
        loads one element and a block has a limited number of threads: the book uses 8 × 8 × 8.
      </p>
      <Predict
        id="ch08/start/p-halo3d"
        prompt={
          <>
            A block loads an 8 × 8 × 8 input tile for the 7-point stencil (one neighbour along each axis). What share of
            the loaded elements is halo? Commit a percentage.
          </>
        }
        unit="%"
      >
        {(v) => <HaloReveal v={v} />}
      </Predict>
    </>
  );
}
