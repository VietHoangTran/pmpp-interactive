import type { Question } from '../../components/Ask';
import { Aside, Source, Today, type Tier } from '../../components/notes';
import {
  BackendsAndEmulator,
  LabFiles,
  LabPrediction,
  LabResults,
  LabsIntro,
  LabsPage,
  ReferenceSolution,
  RunningLabs,
  VerificationStatus,
  benchRow,
  defineLabs,
  measure,
  msCol,
  rawCol,
  useLabs,
  type BenchSet,
} from '../../components/LabKit';
import { fmt, labStatus, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 21's own content.
// No question on this page: the lab is the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch21', [{ num: '21.1', dir: 'lab1_dcs', name: 'scatter, gather, coarsened' }]);
const [l1] = LABS;

/** Identifier colours in the reference solution: where each value lives. */
const TIERS: Record<string, Tier> = {
  atoms: 'const',
  energygrid: 'global',
  energy: 'reg',
  energy0: 'reg',
  energy1: 'reg',
  energy2: 'reg',
  energy3: 'reg',
  charge: 'reg',
  dysqdzsq: 'reg',
  dx: 'reg',
  dy: 'reg',
  dz: 'reg',
  dx0: 'reg',
  dx1: 'reg',
  dx2: 'reg',
  dx3: 'reg',
  dy2: 'reg',
  dz2: 'reg',
  i: 'reg',
  j: 'reg',
  n: 'reg',
  x: 'reg',
  y: 'reg',
  z: 'reg',
};

// ---------------------------------------------------------------- what each prediction is measured against
// Comparison A: a 256 × 256 slice, chunk 256. Comparison B: a 1024 × 1024 slice at four chunk sizes.
const A = { nx: 256, chunk: 256 };
const B_NX = 1024;
const CHUNKS = [64, 256, 512, 1024];
const ms = (r: BenchRow | undefined) => {
  const v = num(r?.seconds);
  return v ? `${fmt(v * 1000)} ms` : '?';
};
const secs = (r: BenchRow | undefined) => num(r?.seconds) ?? 0;

/** Gather over scatter on comparison A. */
const gatherOverScatter = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const sc = benchRow(s, 'scatter', A);
    const ga = benchRow(s, 'gather', A);
    if (!secs(sc) || !secs(ga)) return null;
    return { value: secs(sc) / secs(ga), detail: `scatter ${ms(sc)} against gather ${ms(ga)} per slice` };
  });

/** Adjacent over strided at one chunk size of comparison B: above 1 means the strided kernel was faster. */
const stridedGain = (s: BenchSet, chunk: number) => {
  const adj = benchRow(s, 'coarse_adjacent', { nx: B_NX, chunk });
  const str = benchRow(s, 'coarse_strided', { nx: B_NX, chunk });
  return secs(adj) && secs(str) ? { adj, str, ratio: secs(adj) / secs(str) } : null;
};

/** Strided over adjacent at chunk 64, with the other chunk sizes in the detail. */
const stridedOverAdjacent = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const first = stridedGain(s, CHUNKS[0]);
    if (!first) return null;
    const rest = CHUNKS.slice(1).flatMap((c) => {
      const g = stridedGain(s, c);
      return g ? [`${fmt(g.ratio)} at chunk ${c.toLocaleString('en-US')}`] : [];
    });
    return {
      value: first.ratio,
      detail: `strided ${ms(first.str)} against adjacent ${ms(first.adj)} at chunk 64${rest.length ? `; then ${rest.join(', ')}` : ''}`,
    };
  });

const MISSING = (
  <>
    This needs lab 21.1's benchmark rows. Run <code>python3 labs/pmpp.py run 21.1 --bench</code> on a machine with a GPU
    once your kernels pass (or with <code>--solution</code> to time the reference).
  </>
);

// ---------------------------------------------------------------- results table
const KERNEL_LABEL: Record<string, string> = {
  scatter: 'scatter (Fig. 21.5)',
  gather: 'gather (Fig. 21.6)',
  coarse_adjacent: 'adjacent (provided, Fig. 21.8 corrected)',
  coarse_strided: 'strided (Fig. 21.10)',
};
const gpairs = (r: BenchRow) => {
  const v = num(r.pairs_per_s);
  return v == null ? '' : fmt(v / 1e9);
};

/** One line on lab 21.1 on the race backend, which the status table does not show. */
function RaceStatus() {
  const { reports, loading } = useLabs();
  const run = reports.race;
  const line = (variant: 'solution' | 'work') => {
    const rs = (run?.results.filter((x) => x.lab === l1.id && x.variant === variant) ?? []) as (RunResult & { tsan?: unknown[] })[];
    const tests = rs.flatMap((x) => x.tests);
    const reportsFound = rs.reduce((n, x) => n + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0);
    return { s: labStatus(run, l1.id, variant), tests, passed: tests.filter((t) => t.pass).length, reportsFound };
  };
  const sol = line('solution');
  const mine = line('work');
  return (
    <p data-race-status={loading ? 'loading' : sol.s} data-race-work={loading ? 'loading' : mine.s}>
      {loading ? (
        'Reading reports.'
      ) : !run ? (
        'No run on the race backend on this machine yet.'
      ) : (
        <>
          {sol.s === 'none'
            ? 'On this machine the reference solution has no run on the race backend yet.'
            : sol.s === 'pass'
              ? `On this machine the reference solution passed all ${sol.tests.length} tests on the race backend with ${sol.reportsFound ? `${sol.reportsFound} ThreadSanitizer report(s)` : 'no race reported'}: emulator-checked on this flavour.`
              : `On this machine the reference solution failed on the race backend (${sol.passed} of ${sol.tests.length} tests): unverified there.`}{' '}
          {mine.s === 'none'
            ? 'Your own kernels have no recorded run there, or are still the untouched starter.'
            : mine.s === 'pass'
              ? `Your own kernels passed all ${mine.tests.length} tests there with ${mine.reportsFound ? `${mine.reportsFound} ThreadSanitizer report(s): read them` : 'no race reported'}.`
              : `Your own kernels passed ${mine.passed} of ${mine.tests.length} tests there, with ${mine.reportsFound} ThreadSanitizer report(s) in the run.`}
        </>
      )}
    </p>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS} tiers={TIERS}>
      <LabsIntro>
        One lab, in <code>labs/ch21/lab1_dcs/</code>. It is optional, as this chapter is condensed, and it adds nothing
        to the through-line project: it builds the scatter, gather and coarsened kernels of{' '}
        <a href={href('ch21', '21.2')}>page 21.2</a> for one slice of the potential map and measures the two choices that
        page argues about.
      </LabsIntro>
      <RunningLabs
        example="21.1"
        alsoShownIn={
          <>
            the "yours" group width of the write-statement explorer on <a href={href('ch21', '21.2')}>21.2</a>
          </>
        }
      />
      <BackendsAndEmulator />
      <VerificationStatus />

      {/* ------------------------------------------------------------ 21.1 */}
      <h2>Lab 21.1: one slice, three ways</h2>
      <p>
        Every point of a slice of an <code>nz × ny × nx</code> grid gets the sum, over all atoms, of charge divided by
        distance. The atoms reach the device in chunks small enough for constant memory, and each launch adds one chunk's
        contribution to the slice. You write three kernels and their launch functions. The scatter kernel of Fig. 21.5
        gives each thread one atom of the chunk, walks the whole slice and adds with a float atomic. The gather kernel of
        Fig. 21.6 gives each thread one point, sums the chunk in a register and adds the sum with one plain{' '}
        <code>+=</code>. The coarsened kernel of Fig. 21.10 gives each thread four points of a row,{' '}
        <code>blockDim.x</code> apart, reads each atom once and shares <code>dy*dy + dz*dz</code> and the charge across
        the four.
      </p>
      <Source section="21.2 to 21.4" pages="515–522" figs="Figs. 21.5, 21.6, 21.10; exercise 1" />
      <p>
        Each launch function is the host loop of p. 517, which is the book's exercise 1: copy at most{' '}
        <code>chunk</code> atoms into the <code>__constant__</code> array with <code>cudaMemcpyToSymbol</code>, launch,
        repeat until every atom has been added; the last chunk is usually partial. The contract is in{' '}
        <code>dcs.h</code>: the grid is a row-major device pointer with x fastest, the atoms are four floats each (x, y,
        z, charge) in host memory, a launch adds to slice <code>k</code> and leaves the others alone, and{' '}
        <code>CHUNK_MAX</code>, <code>BLOCK_X</code>, <code>BLOCK_Y</code> and <code>COARSEN</code> are defined there
        and nowhere else.
      </p>
      <p>
        One kernel is provided and not edited: <code>labs/ch21/shared/coarse_adjacent.cu</code> (with its Metal pair),
        Fig. 21.8 with its line 05 corrected, so each thread owns four adjacent points. Your strided kernel is timed
        against it. It keeps its own constant array, <code>atoms_adj</code>: under the emulator a{' '}
        <code>__constant__</code> variable is an ordinary global, and two of the same name would collide when the files
        are linked.
      </p>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <>
              the <code>__constant__</code> atom array, <code>dcsScatterKernel</code>, <code>dcsGatherKernel</code>,{' '}
              <code>dcsCoarsenedKernel</code> and their launches <code>launchDcsScatter</code>,{' '}
              <code>launchDcsGather</code> and <code>launchDcsCoarsened</code>
            </>,
          ],
          [
            'kernel.metal',
            <>
              <code>dcs_scatter</code>, <code>dcs_gather</code> and <code>dcs_coarsened</code>, with the atoms as{' '}
              <code>constant float4*</code>
            </>,
          ],
          [
            'launch.mm',
            <>
              the three chunk loops, in <code>encodeDcsScatter</code>, <code>encodeDcsGather</code> and{' '}
              <code>encodeDcsCoarsened</code>
            </>,
          ],
        ]}
      />
      <p>Where the lab differs from the printed listings, each difference tested:</p>
      <ul>
        <li>
          Fig. 21.8's line 05 adds an unscaled <code>threadIdx.x</code> while each thread writes <code>i</code> to{' '}
          <code>i+3</code>, so neighbouring threads overlap and most of each block's points are never written (
          <a href={href('ch21', '21.2')}>21.2</a>, the erratum hunt). The provided baseline scales the index by the factor,
          as the text on p. 521 describes. The same line 05 is right in Fig. 21.10, whose ownership is strided, and your
          kernel keeps it.
        </li>
        <li>
          The slice is an <code>int k</code> and its z is <code>k * gridspacing</code>. Fig. 21.6 passes a float{' '}
          <code>z</code> and computes <code>k = z / gridspacing</code>, which truncates, and Figs. 21.3 and 21.4 index the
          slice two other ways. The sizes are <code>nx</code> and <code>ny</code> where the book passes a{' '}
          <code>dim3 grid</code>.
        </li>
        <li>
          No listing tests a bound, and Fig. 21.5 has no atom count at all: the threads past the end of a partial chunk
          would add atoms the previous chunk left in constant memory. Here the grid is not padded to whole blocks, so every
          kernel tests its bounds, the coarsened one once per point.
        </li>
        <li>
          The chapter does not name the call that fills constant memory; it is <code>cudaMemcpyToSymbol</code>. The
          book's float <code>atomicAdd</code> is kept: NVIDIA now calls it a legacy atomic and recommends{' '}
          <code>cuda::atomic_ref</code>, which chapter 9 used.
        </li>
      </ul>
      <p>
        Run <code>python3 labs/pmpp.py run 21.1</code>, starting with <code>--backend emu</code>. Every case runs a whole
        slice through your launch function and compares each point with a sum in double in <code>main.cpp</code>, within{' '}
        <code>1e-4</code> of the sum over atoms of |charge| / distance at that point. That allows for Metal's fast math and
        for float atomics adding in no fixed order; no comparison is bitwise. The slices are 37 × 11 and 300 × 5, neither
        a multiple of 4 × <code>BLOCK_X</code>; 150 atoms go in three launches of 64, 64 and 22, and also in one launch,
        and one case has a single atom. Slice 1 or 2 of 3 is computed while the other slices hold a sentinel that must
        survive bit for bit, and slice 2 is the last, so a write past a row's end leaves the allocation. Every atom sits
        half-way between two slice planes, so no distance is zero. A GPU also runs 1000 × 333 with 2,000 atoms in chunks
        of 1,024. All four kernels run every case, and gather and coarsened must also agree with each other.
      </p>
      <p>
        What each backend can catch matters here more than in most labs. The emulator runs one thread at a time and never
        interrupts a <code>+=</code> half-way, so it cannot see a missing atomic: a scatter kernel with a plain{' '}
        <code>+=</code> passes there. It does catch a wrong index, a missing bound (AddressSanitizer names the line), stale
        atoms from a missing atom count, <code>=</code> where <code>+=</code> belongs (the three-launch cases) and a
        missing slice offset (the sentinel cases). The race flavour,{' '}
        <code>python3 labs/pmpp.py run 21.1 --backend race</code>, runs the threads of a block concurrently under
        ThreadSanitizer on small slices: with the atomic replaced by <code>+=</code> it reported the scatter kernel's
        write line, and with Fig. 21.8's printed line 05 it reported the four overlapping writes. A clean run is evidence,
        not proof: it sees only the schedules that happened. On a GPU a missing atomic shows only as wrong sums, and only
        when two threads collided in time.
      </p>
      <RaceStatus />
      <p>
        One gap the tests leave open: the 37-wide cases cannot see a wrong <code>dx1</code>, <code>dx2</code> or{' '}
        <code>dx3</code> in your coarsened kernel, because a 37-point row holds only each thread's first point. The
        300-wide cases can, so a kernel that passes the first and fails the second has its offsets and its write
        positions out of step.
      </p>
      <p>
        On Metal there is no symbol to copy to. All the atoms sit in one buffer of <code>float4</code>s, and a chunk is
        that buffer bound again at the chunk's first atom, 16 bytes per atom. The scatter kernel declares the grid{' '}
        <code>device atomic_float*</code> and adds with <code>atomic_fetch_add_explicit</code>; the other two take the same
        buffer as <code>device float*</code>. Launch the coarsened kernel with <code>dispatchThreadgroups</code>, whole
        threadgroups of <code>BLOCK_X × BLOCK_Y</code>, and read its stride from{' '}
        <code>[[dispatch_threads_per_threadgroup]]</code>: <code>dispatchThreads</code> would trim the edge threadgroups,
        and a trimmed group's missing threads own points that nobody computes (
        <a href={href('ch21', '21.2')}>21.2</a> has a question on it).
      </p>
      <Today checked="2026-11-20 (notes/today/ch21-facts.md C1, C2; infra-facts C1, C6; ch05-facts C4, C5; ch02-facts C9)">
        <p>
          Metal: float atomics take an atomic type and need Apple7 (M1) or later. Apple documents no constant cache and no
          size limit behind <code>constant</code> buffers. An edge threadgroup of <code>dispatchThreads</code> is smaller,
          and <code>[[threads_per_threadgroup]]</code> reports its actual size. Dispatches of one compute encoder run in
          order and see each other's writes.
        </p>
      </Today>
      <p>
        The same <code>kernel.cu</code> builds for HIP, where <code>cudaMemcpyToSymbol</code> maps to{' '}
        <code>hipMemcpyToSymbol</code>. The course's HIP build does not pass <code>-munsafe-fp-atomics</code>, so a scatter
        measurement on an AMD GPU is of the safe form of the float atomic.
      </p>
      <Today checked="2026-11-20 (notes/today/infra-facts.md B1, B5; ch21-facts B1)">
        <p>
          HIP compiles float <code>atomicAdd</code> as a safe operation, possibly a compare-and-swap loop, unless built
          with <code>-munsafe-fp-atomics</code>; RDNA1 and CDNA1 have no hardware float atomic add. AMD documents no size
          for constant memory, and its scalar cache helps only reads that are the same across the wavefront.
        </p>
      </Today>

      <h3>The experiment</h3>
      <p>
        Do this once your kernels pass. Comparison A times scatter against gather on a 256 × 256 slice with 1,024 atoms in
        chunks of 256; it is kept small because scatter has one thread per atom and each one walks the whole slice.
        Comparison B times gather, the provided adjacent kernel and your strided kernel on a 1024 × 1024 slice with 4,096
        atoms, at chunks of 64, 256, 512 and 1,024. Every time is one whole slice, all chunks included, the best of five.
        Write down two predictions before you run <code>--bench</code>.
      </p>
      <Source section="21.2, 21.4" pages="518, 521–522" />
      <Aside>
        <p>
          Chunk 512 is 8 KB of atoms, the constant-cache working set per SM that NVIDIA documents (ch21-facts A5). Whether
          anything changes there is not documented; comparison B is where you would see it.
        </p>
      </Aside>
      <LabPrediction
        id="ch21/labs/p-scatter"
        prompt={
          <>
            On comparison A, how many times faster than your scatter kernel will your gather kernel run on your GPU? Think
            about what every thread of a chunk is doing at the same moment.
          </>
        }
        unit="× the scatter kernel"
        measure={gatherOverScatter}
        missing={MISSING}
        reading={
          <>
            The book says only that atomics "significantly" slow the scatter kernel (p. 518), and no vendor gives a ratio.
            Two costs stack up: every lane of a warp adds to the same point at the same step, so the atomics on it are
            serialised, and every lane reads a different atom, the slow case of the constant cache on NVIDIA and AMD.
            Apple documents neither cost, and on AMD the atomic's form depends on a compiler flag.
          </>
        }
      />
      <LabPrediction
        id="ch21/labs/p-strided"
        prompt={
          <>
            On comparison B at chunk 64, how many times faster than the provided adjacent kernel will your strided kernel
            run? Note down too whether you expect that gap to grow or shrink by chunk 1,024; the reveal shows all four
            chunk sizes.
          </>
        }
        unit="× the adjacent kernel"
        measure={stridedOverAdjacent}
        missing={MISSING}
        reading={
          <>
            Above 1, the strided kernel was faster. On NVIDIA the documented rule gives 64 sectors for the adjacent
            kernel's four write statements per warp and 16 for the strided kernel's. But a launch reads and writes each
            point once while its atom loop runs <code>chunk</code> times per point, so our own prediction, unverified, is
            that the gap shrinks as the chunk grows. On Metal and HIP no document predicts either number: Apple documents
            no coalescing rule. Neighbouring sizes can differ by a few per cent between runs; run the benchmark twice before
            you read much into a small gap.
          </>
        }
      />
      <LabResults
        lab={l1}
        cols={[
          { head: 'What ran', cell: (r) => KERNEL_LABEL[r.name] ?? r.name },
          rawCol('Comparison', 'comparison'),
          { head: 'Slice', cell: (r) => `${String(r.nx)} × ${String(r.ny)}` },
          rawCol('Atoms', 'atoms'),
          rawCol('Chunk', 'chunk'),
          msCol,
          { head: 'Gpairs/s', num: true, cell: gpairs },
          rawCol('Group width', 'group_width'),
        ]}
      />
      <p>
        A pair is one atom and one point, so pairs per second compares kernels on slices of different sizes. The group
        width is <code>warpSize</code> on CUDA and HIP and, on Metal, the <code>threadExecutionWidth</code> of the coarsened
        pipeline; the write-statement explorer on <a href={href('ch21', '21.2')}>21.2</a> offers it as yours. On CUDA,
        Nsight Compute's sectors per request for the global stores of your kernel and of the adjacent one shows the
        difference the model predicts; for 32-bit accesses by a full warp the optimum is 4 (ch21-facts A8). Nothing here
        ran on a CUDA device.
      </p>
      <p>
        To go further: PyTorch makes this lab's choice for the gradient of an embedding table, differently on two backends.
        On CUDA and ROCm it sorts the token ids and sums each one's rows with a single writer, unless a heuristic prefers its
        atomic path and deterministic algorithms were not requested; on MPS it adds with float atomics and refuses in
        deterministic mode (ch21-facts D1, D2). From what you measured in comparison A, which would you ship, and when?
      </p>
      <ReferenceSolution lab={l1} />
    </LabsPage>
  );
}
