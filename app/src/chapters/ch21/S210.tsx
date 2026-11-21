import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { SEGMENT_MODELS, warpAccess } from '../../sim/coalesce';

// The start page of chapter 21: what the chapter adds (four decisions about one all-pairs
// sum), the domain in one paragraph, one bridge to PyTorch with a prediction, a map, and a
// check on the five ideas from chapters 4, 6, 7, 9 and appendix A that the chapter uses.
// The questions come in the order the chapter uses them. The sector count comes from the
// tested coalescing model (src/sim/coalesce.ts); the matrix size is plain arithmetic.

// The sector question is about NVIDIA only, where a warp is 32 threads
// (notes/today/ch05-facts.md A5) and the unit is a 32-byte sector (ch06-facts A2).
const NVIDIA_WARP = 32;
const SECTOR = SEGMENT_MODELS.nvidia.segmentBytes ?? 32;
const F32 = 4;
const lanes = Array.from({ length: NVIDIA_WARP }, (_, t) => t);
/** Lane t stores element base + 4t: the stride of one write statement in Fig. 21.8 as the text describes it. */
const SECTORS_STRIDE4 = warpAccess(lanes.map((t) => 4 * t * F32), F32, SECTOR).transactions; // 16
/** Lane t stores element base + t: Fig. 21.10. */
const SECTORS_STRIDE1 = warpAccess(lanes.map((t) => t * F32), F32, SECTOR).transactions; // 4

// The prediction: the scale is the book's (notes/ch22.md, p. 536: hundreds of thousands of
// atoms, millions of grid points); the round numbers are ours.
const POINTS = 1_000_000;
const ATOMS = 100_000;
const BYTES = POINTS * ATOMS * F32;
const MATRIX_GB = BYTES / 1e9; // 400
const MATRIX_GIB = BYTES / 2 ** 30; // 372.5
const RESULT_MB = (POINTS * F32) / 1e6; // 4
const fmt = (v: number) => (Math.round(v * 10) / 10).toLocaleString('en-US');

export const questionsS210: Record<string, Question> = {
  'ch21/start/q-float-order': {
    kind: 'choice',
    id: 'ch21/start/q-float-order',
    prompt: (
      <>
        A kernel adds 10,000 float contributions into one output location with <code>atomicAdd</code>. You run it
        twice. Both results agree with a double-precision CPU reference to within your test's tolerance, but they differ
        from each other in the last few digits. Is that a bug?
      </>
    ),
    options: [
      {
        label: (
          <>
            No. Same-address atomics land in no fixed order and float addition is not associative, so each run rounds
            differently.
          </>
        ),
        correct: true,
      },
      {
        label: 'Yes. Some updates are being lost: when two lanes of a warp hit one address in the same instruction, the hardware keeps only one value.',
        why: 'Not losing updates is exactly what an atomic guarantees. A lost update drops a whole contribution, and the sum would then miss the reference, not just the other run.',
      },
      {
        label: 'Yes. A correct kernel run twice on the same inputs gives the same bits, so any difference means a race somewhere.',
        why: 'Same bits is reproducibility, not correctness. Both runs match the reference; ordinary rounded additions, done in a different order, already give different last digits.',
      },
      {
        label: 'No, and it cannot happen: the GPU applies atomics in thread-index order, so the difference must come from the host.',
        why: 'NVIDIA says the order of atomics on one address is undefined, and the book says atomics enforce no order between threads.',
      },
    ],
    explain: (
      <p>
        The book says atomics enforce no order between threads; each one only waits for the one before it to finish.
        NVIDIA adds that when several threads of a warp update one location the updates are serialised "but the order
        in which they occur is undefined". Appendix A shows that the order of float additions changes the rounded sum.
        Together: a sum built from float atomics is correct to rounding but not bit-for-bit reproducible, while a thread
        that adds its terms in a fixed order gets the same answer every time. Page{' '}
        <a href={href('ch21', '21.2')}>21.2</a> meets both: the scatter kernel adds into the grid with{' '}
        <code>atomicAdd</code>, the gather kernel sums in a register, and PyTorch draws the same line when you ask it
        for deterministic algorithms. Taught in <a href={href('ch09', '9.2')}>ch. 9 §9.2</a> and appendix A §A.6.
      </p>
    ),
    source:
      'ch. 9 §9.2, pp. 203–209 (no order: p. 206); appendix A §A.6, pp. 593–594; notes/today/infra-facts.md A1; ch10-facts A9',
  },
  'ch21/start/q-const-lanes': {
    kind: 'choice',
    id: 'ch21/start/q-const-lanes',
    prompt: (
      <>
        A table sits in <code>__constant__</code> memory. In kernel A every lane of a warp reads <code>table[n]</code>{' '}
        with the same <code>n</code>. In kernel B lane <code>t</code> reads <code>table[t]</code>. What does NVIDIA
        document about the two reads?
      </>
    ),
    options: [
      {
        label: (
          <>
            A: one access for the whole warp, which can be as fast as a register. B: one access per distinct address,
            so one per lane.
          </>
        ),
        correct: true,
      },
      {
        label: 'Both are broadcast to the warp in one access, because constant data is read-only and needs no coherence traffic.',
        why: 'Being read-only is what lets the cache be small and cheap. The cost is still one access per distinct address in the warp.',
      },
      {
        label: 'B is faster: like shared memory, the constant cache has banks, and 32 distinct addresses spread over them with no conflict.',
        why: 'That is the rule for shared memory. The constant cache serves distinct addresses one after another.',
      },
      {
        label: 'A and B: one cache access per warp each once the table is in the cache; only a miss makes B dearer.',
        why: 'Even when every read hits, distinct addresses in one warp are served one at a time.',
      },
    ],
    explain: (
      <p>
        NVIDIA: "Accesses to different addresses by threads within a warp are serialized, thus the cost scales linearly
        with the number of unique addresses", and when every thread of a warp reads one location, constant memory "can
        be as fast as a register access". AMD documents the same split for its scalar cache, which serves only reads
        that are uniform across the wavefront; Apple documents no constant cache at all. On page{' '}
        <a href={href('ch21', '21.2')}>21.2</a> the atoms sit in constant memory. In the gather kernel every lane of a
        warp reads the same atom at the same time: case A. In the scatter kernel each thread reads its own atom: case B.
        Taught in <a href={href('ch07', '7.4')}>ch. 7 §7.4</a>.
      </p>
    ),
    source: 'ch. 7 §7.4, pp. 168–172; notes/today/ch07-facts.md A3 (NVIDIA), C2 (Apple); ch21-facts B1 (AMD)',
  },
  'ch21/start/q-coarsen-cost': {
    kind: 'choice',
    id: 'ch21/start/q-coarsen-cost',
    prompt: (
      <>
        Each thread of a kernel computes one output by looping over a read-only list that every thread reads. You
        coarsen it: each thread now computes four outputs, loads each list item once and uses it four times, and you
        launch a quarter of the threads. The problem fills the GPU many times over. What can still make the coarsened
        kernel slower?
      </>
    ),
    options: [
      {
        label: 'Each thread keeps more values live in registers, so fewer warps fit on an SM to hide memory latency.',
        correct: true,
      },
      {
        label: 'Each output now needs more arithmetic than before, because the thread must also compute the offsets of its four outputs.',
        why: 'Coarsening removes repeated work. Per output, the thread does the same work or less.',
      },
      {
        label: 'The four outputs of a thread need a barrier between them, so the thread waits for its warp at every output.',
        why: 'One thread computes all four in sequence. There is no other thread to wait for.',
      },
      {
        label: 'Too few blocks are left to keep every SM busy, since a quarter of the threads means a quarter of the blocks.',
        why: "That is the book's second pitfall, too much coarsening, and it bites when the problem is small. This one fills the device many times over.",
      },
    ],
    explain: (
      <p>
        The book's third pitfall: coarsening can raise the registers each thread needs, and if that lowers occupancy the
        loss can outweigh the gain. The register file of an SM is fixed, so heavier threads mean fewer resident warps to
        switch between while others wait on memory. Which effect wins is a measurement, not a rule. The coarsened kernel
        on page <a href={href('ch21', '21.2')}>21.2</a> keeps four running sums, four distances and the terms the four
        points share live for the whole atom loop. The book says its register count stays within the limit but gives no
        count; you check your own build with <code>--resource-usage</code> under <code>nvcc</code> or{' '}
        <code>hipcc</code>, and Metal reports no register count. Taught in{' '}
        <a href={href('ch06', '6.5')}>ch. 6 §6.5</a>.
      </p>
    ),
    source: 'ch. 6 §6.5, pp. 141–143 (pitfall 3: p. 143); ch. 21 §21.3, p. 521; notes/today/ch21-facts.md A7',
  },
  'ch21/start/q-sectors': {
    kind: 'numeric',
    id: 'ch21/start/q-sectors',
    prompt: (
      <>
        On an NVIDIA GPU a warp has 32 lanes. Each lane stores one float: lane <code>t</code> writes element{' '}
        <code>base + 4*t</code>, and element <code>base</code> starts on a 128-byte boundary. How many 32-byte sectors
        does the warp's store touch?
      </>
    ),
    answer: SECTORS_STRIDE4,
    unit: 'sectors',
    explain: (
      <p>
        The lanes cover elements 0 to 124 of a 128-element stretch, 512 bytes. Every 32-byte sector in it holds two of
        the stored floats and six that nobody asked for: {SECTORS_STRIDE4} sectors for 128 useful bytes. With lane{' '}
        <code>t</code> writing <code>base + t</code>, the same 32 floats fill {SECTORS_STRIDE1} sectors exactly. The
        32-byte sector is NVIDIA's unit; AMD groups a wavefront's addresses into cache lines, and Apple documents no
        rule. On page <a href={href('ch21', '21.2')}>21.2</a> a coarsened thread owns four adjacent grid points, so
        each of its four write statements has this stride of four across the warp: {4 * SECTORS_STRIDE4} sectors for
        the four, where the book's fix needs {4 * SECTORS_STRIDE1}. Taught in{' '}
        <a href={href('ch06', '6.1')}>ch. 6 §6.1</a>.
      </p>
    ),
    source: 'ch. 6 §6.1, pp. 124–132; notes/today/ch06-facts.md A2 (NVIDIA\'s unit), B1, C1; ch21-facts E2',
  },
  'ch21/start/q-divergent-test': {
    kind: 'choice',
    id: 'ch21/start/q-divergent-test',
    prompt: (
      <>
        A warp of 32 lanes runs <code>if (r2 &lt; c2) e += q / sqrtf(r2);</code> and 8 of its lanes pass the test.
        What does the warp execute?
      </>
    ),
    options: [
      {
        label:
          'The warp runs the body once with 8 lanes active and 24 masked off, taking about as long as if all 32 passed.',
        correct: true,
      },
      {
        label: 'The body 8 times, once for each passing lane, so it costs a quarter of what 32 passing lanes would.',
        why: 'A warp runs one instruction for all its active lanes at once. The passes are per path, not per lane.',
      },
      {
        label: 'Nothing: a warp skips a branch that most of its lanes do not take, and the 8 passing lanes keep their old sums.',
        why: 'A warp skips a path only when none of its lanes take it. One lane is enough for the whole warp to run the body.',
      },
      {
        label: 'The body once, in about a quarter of the time: only 8 of the 32 lanes are active.',
        why: 'The masked lanes still occupy their slots for the whole pass. The book counts that as one of the two costs of divergence.',
      },
    ],
    explain: (
      <p>
        NVIDIA: the warp "executes each branch path taken, disabling threads that are not on that path". Here there is
        one path, the body, and the 24 failing lanes sit idle through it. NVIDIA gives no figure for what a pass costs,
        and notes that a compiler may turn a short body like this into predicated instructions so that no warp diverges
        at all; neither description makes the body cheaper when fewer lanes need it. On page{' '}
        <a href={href('ch21', '21.5')}>21.5</a> every thread tests every atom of a neighbourhood bin against the cutoff
        around its own grid point, and at the book's worked numbers about three tests in four fail (our count). Taught
        in <a href={href('ch04', '4.5')}>ch. 4 §4.5</a>.
      </p>
    ),
    source:
      'ch. 4 §4.5, pp. 79–82; notes/today/ch04-facts.md A9; ch06-facts A14 (predication); ch21-facts E4 (our count at the p. 525 numbers)',
  },
};

const near = (v: number, x: number, rel: number) => Math.abs(v - x) <= Math.abs(x) * rel;

/** The reveal after the matrix-size prediction, written for the common slips. */
function MaterialiseReveal({ v }: { v: number }) {
  const right = near(v, MATRIX_GB, 0.1) || near(v, MATRIX_GIB, 0.05);
  const gib = right && Math.abs(v - MATRIX_GIB) < Math.abs(v - MATRIX_GB);
  const noBytes = near(v, MATRIX_GB / F32, 0.1); // 100: elements, not bytes
  const fewPoints = near(v, MATRIX_GB / 10, 0.1); // 40: 10^5 points
  const twoMatrices = near(v, 2 * MATRIX_GB, 0.1); // 800: distances and quotients
  const terabytes = near(v, MATRIX_GB / 1000, 0.1); // 0.4: TB
  const decades = v > 0 ? Math.log10(v / MATRIX_GB) : NaN;
  const powerOfTen = !isNaN(decades) && Math.round(decades) !== 0 && Math.abs(decades - Math.round(decades)) < 0.05;
  return (
    <>
      <p>
        <b>
          10<sup>6</sup> points × 10<sup>5</sup> atoms × 4 bytes = 4 × 10<sup>11</sup> bytes: {fmt(MATRIX_GB)} GB.
        </b>{' '}
        That is {fmt(MATRIX_GIB)} GiB if you count in powers of two.
      </p>
      {right ? (
        <p>
          {gib ? 'That is what you predicted, in GiB. ' : 'That is what you predicted. '}
          Every pair gets its own entry, and there are 10<sup>11</sup> pairs.
        </p>
      ) : noBytes ? (
        <p>
          You counted entries, not bytes: 10<sup>11</sup> entries is 100 billion. Each entry is a float32 of 4 bytes, so
          the matrix is four times that in bytes.
        </p>
      ) : fewPoints ? (
        <p>
          That is 10<sup>5</sup> × 10<sup>5</sup> × 4 bytes. The grid has ten times more points than there are atoms:
          10<sup>6</sup>.
        </p>
      ) : twoMatrices ? (
        <p>
          You counted two matrices, the distances and the quotients. Written this way the expression can indeed create
          both; the question asked for one.
        </p>
      ) : terabytes ? (
        <p>That is the size in terabytes. In gigabytes it is a thousand times the number.</p>
      ) : powerOfTen ? (
        <p>
          You are off by a power of ten. Count the zeros: 6 for the points, 5 for the atoms, then multiply by 4 bytes
          and divide by 10<sup>9</sup> for gigabytes.
        </p>
      ) : (
        <p>
          The count is one entry per pair: 10<sup>6</sup> × 10<sup>5</sup> = 10<sup>11</sup> entries, 4 bytes each.
        </p>
      )}
      <p>
        The answer it reduces to is one float per grid point, {fmt(RESULT_MB)} MB. To get there the matrix has to be
        written at least once and read back at least once: at least {fmt(2 * MATRIX_GB)} GB of memory traffic for a{' '}
        {fmt(RESULT_MB)} MB result, before any question of whether it fits.
      </p>
    </>
  );
}

export function S210() {
  const q = questionsS210;
  return (
    <>
      <p>
        This chapter takes one computation through a sequence of kernels: the electrostatic potential at every point of
        a 3D grid, caused by a set of charged atoms. The book calls the method direct Coulomb summation, DCS for short.
        Molecules play no part in LLM inference, and the domain gets one paragraph here. What transfers is the sequence
        of four decisions the book makes about one all-pairs sum:
      </p>
      <Source section="21 (opening), 21.6" pages="513, 527–528" />
      <ul>
        <li>
          <b>Who owns each output.</b> A thread per atom scatters its contributions over the grid and needs an atomic
          for every update. A thread per grid point gathers from every atom and writes once (§21.2, page{' '}
          <a href={href('ch21', '21.2')}>21.2</a>).
        </li>
        <li>
          <b>Reuse inside one thread.</b> Coarsening: one thread computes several grid points, loads each atom once and
          reuses what the points share (§21.3, page <a href={href('ch21', '21.2')}>21.2</a>).
        </li>
        <li>
          <b>Which outputs a thread owns.</b> Giving a thread adjacent points makes every write statement uncoalesced;
          spacing them a block's width apart fixes it (§21.4, page <a href={href('ch21', '21.2')}>21.2</a>).
        </li>
        <li>
          <b>Changing the algorithm so it scales.</b> All-pairs work grows with the square of the system's volume. A
          cutoff with spatial bins makes it grow linearly, at a small cost in accuracy (§21.5, page{' '}
          <a href={href('ch21', '21.5')}>21.5</a>).
        </li>
      </ul>
      <Aside>
        <p>
          The opening (p. 513) lists a "gather-to-scatter transformation" among the chapter's techniques. The chapter
          goes the other way, from scatter to gather, and recommends gather. The same list names "fast boundary
          condition checking", which no section develops.
        </p>
      </Aside>

      <h2>The problem in one paragraph</h2>
      <p>
        VMD is a program for displaying, animating and analysing models of biomolecules. One thing it computes is an
        electrostatic potential map, used for example to find where ions fit around a molecule before a
        molecular-dynamics simulation.
      </p>
      <Source section="21.1" pages="513–514" figs="Fig. 21.2, p. 515" />
      <p>
        The potential at grid point <i>j</i> is a sum over all atoms of each atom's charge divided by its distance to
        the point:{' '}
        <span style={{ whiteSpace: 'nowrap' }}>
          <i>V</i>
          <sub>j</sub> = Σ<sub>i</sub> <i>q</i>
          <sub>i</sub> / <i>r</i>
          <sub>ij</sub>
        </span>
        . Every atom contributes to every point, so the work is the number of atoms times the number of
        grid points. That is why the computation has traditionally run as a batch job, and why making it fast enough to
        use interactively was worth a GPU.
      </p>
      <Today checked="2026-11-20 (notes/today/ch21-facts.md, A1)">
        <p>
          VMD is still released: version 2.0.0 came out on 2026-03-25, with a Linux build that uses CUDA. Its GPU page
          lists ion placement and Coulomb summation among the GPU-accelerated features.
        </p>
      </Today>

      <h2>In PyTorch terms</h2>
      <p>
        With the grid points as a <code>(P, 3)</code> tensor, the atom positions as <code>(A, 3)</code> and the charges
        as <code>(A,)</code>, one line computes the map: <code>V = (q / torch.cdist(points, atoms)).sum(-1)</code>. It
        is correct, and it builds a P{' '}×{' '}A matrix of distances before it reduces anything. Chapter 22 gives the scale of
        a realistic system: hundreds of thousands of atoms and millions of grid points.
      </p>
      <Beyond>
        <p>
          The PyTorch line and the comparison with attention are ours. The scale is the book's (ch. 22 §22.2, p. 536),
          as is the fact that FlashAttention never writes its score matrix to global memory (ch. 20 §20.5, p. 497).
        </p>
      </Beyond>
      <Predict
        id="ch21/start/p-materialise"
        prompt={
          <>
            Take 1,000,000 grid points and 100,000 atoms. How large is the float32 matrix that{' '}
            <code>q / torch.cdist(points, atoms)</code> builds before <code>.sum(-1)</code> reduces it? Commit a size
            in GB.
          </>
        }
        unit="GB"
      >
        {(v) => <MaterialiseReveal v={v} />}
      </Predict>
      <p>
        No kernel in this chapter forms that matrix. Each pair's contribution is computed and added straight into a
        sum: into a register in the gather kernels, into the grid through an atomic in the scatter kernel. A fused
        attention kernel avoids writing its score matrix for the same reason, though softmax makes it harder there:
        chapter 20 builds one (<a href={href('ch20', '20.5')}>§20.5</a>).
      </p>

      <h2>The chapter in this course</h2>
      <p>
        Two pages. Page <a href={href('ch21', '21.2')}>21.2</a> covers the book's §§21.2 to 21.4: the sequential code
        and its loop interchange, the scatter kernel with atomics, the gather kernel with its atoms in constant memory,
        coarsening by four, and the change of ownership that coalesces the writes (pp. 515–522). Page{' '}
        <a href={href('ch21', '21.5')}>21.5</a> covers §21.5: why direct summation does not scale, the cutoff, spatial
        bins and the neighbourhood list each block scans, and fixed-size bins padded with dummy atoms, with an overflow
        list the host finishes (pp. 523–527). Then one optional <a href={href('ch21', 'labs')}>lab</a>, which builds
        the scatter, gather and coarsened kernels and runs them on your GPU, and the{' '}
        <a href={href('ch21', 'wrap')}>summary</a> with the book's exercises and the PDF's apparent slips.
      </p>
      <p>
        What is condensed: the background of §21.1 (pp. 513–514), with VMD's list of accelerated features and the
        rendered molecule of Fig. 21.1, is reduced to the paragraph above. The summary page lists what else is left
        out, with pages.
      </p>
      <p>
        The first decision comes back in chapter 22. Its <a href={href('ch22', '22.2')}>§22.2</a> sorts every pattern
        in the book by it, output-centric (gather) or input-centric (scatter), and uses this chapter as the case where
        gather wins because it needs no atomics.
      </p>
      <Source section="22.2" pages="532–536" />

      <h2>Before you start</h2>
      <p>
        Five questions on ideas from chapters 4, 6, 7 and 9 and appendix A, in the order this chapter uses them. Each
        explanation ends with the page that relies on it, and each one comes back in{' '}
        <a href={href('review')}>review</a>.
      </p>
      <Ask q={q['ch21/start/q-float-order']} label="From chapter 9 and appendix A" />
      <Ask q={q['ch21/start/q-const-lanes']} label="From chapter 7" />
      <Ask q={q['ch21/start/q-coarsen-cost']} label="From chapter 6" />
      <Ask q={q['ch21/start/q-sectors']} label="From chapter 6" />
      <Ask q={q['ch21/start/q-divergent-test']} label="From chapter 4" />
    </>
  );
}
