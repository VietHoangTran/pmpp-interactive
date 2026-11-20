import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Beyond, Source } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { benchRows, fmt, labStatus, num, realBackends, useReports, type BackendRun, type BenchRow } from '../../lib/reports';
import { href } from '../../lib/router';
import { warpAccess } from '../../sim/coalesce';
import { acceptedShare, counts, superRadius } from '../../sim/cutoff';
import { arithmetic, opsGather } from '../../sim/dcs';

// A kernel of our own, for one question. Checked copy: labs/ch21/snippets/bias_gather.cu
// (Wrap.snippets.test.ts ties the two and pins the numbers the other questions take from
// the models).
const BIAS = `// Ours: a logit bias in the gather form of Fig. 21.6. The list holds n
// (token id, bias) pairs, and a token may appear in it more than once.
__global__ void biasGather(float* logits, int vocab,
                           const int* ids, const float* bias, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's logit
    if (v >= vocab) return;
    float add = 0.0f;
    for (int e = 0; e < n; ++e)   // every thread scans every pair
        if (ids[e] == v) add += bias[e];
    logits[v] += add;             // one writer per logit: no atomic
}`;

// ---------------------------------------------------------------- the numbers the questions take from the models
/** The bias question's sizes (ours). */
export const VOCAB = 128_000;
export const PAIRS = 50;

/** Gather kernel, per point per launch: arithmetic per atom (dcs.ts, counted from Fig. 21.6), over 8 bytes of +=. */
export const OPS_PER_ATOM = arithmetic(opsGather(1)); // 10
export const CHUNK = 64;
export const RMW_BYTES = 8; // one float read and one written by the +=
export const INTENSITY = (CHUNK * OPS_PER_ATOM) / RMW_BYTES; // 80
export const INTENSITY_1024 = (1024 * OPS_PER_ATOM) / RMW_BYTES; // 1,280

/** Coarsening by 8 on an NVIDIA warp: sectors per write statement (warpAccess, chapter 6's model). */
const NVIDIA_WARP = 32;
const SECTOR = 32;
const F32 = 4;
export const FACTOR = 8;
const lanes = Array.from({ length: NVIDIA_WARP }, (_, t) => t);
export const SECTORS_ADJ_STMT = warpAccess(lanes.map((t) => FACTOR * t * F32), F32, SECTOR).transactions; // 32
export const SECTORS_STR_STMT = warpAccess(lanes.map((t) => t * F32), F32, SECTOR).transactions; // 4
export const SECTORS_ADJ_ALL = FACTOR * SECTORS_ADJ_STMT; // 256
export const SECTORS_STR_ALL = FACTOR * SECTORS_STR_STMT; // 32
/** Factor 4, for comparison: page 21.2's 16 per statement. */
export const SECTORS_ADJ_F4 = warpAccess(lanes.map((t) => 4 * t * F32), F32, SECTOR).transactions; // 16

/** The book's worked case (p. 525), 3D: 12 Å cutoff, 4 Å bins; then 2 Å bins. */
const CUTOFF = 12;
const listFor = (b: number) => counts(superRadius(CUTOFF, b, 3, 'half') / b, 3).total;
const pct = (v: number) => Math.round(100 * v);
export const BINS_4 = listFor(4); // 461
export const BINS_2 = listFor(2); // 1,863
export const SHARE_4 = pct(acceptedShare(CUTOFF, 4, 3)); // 25
export const SHARE_2 = pct(acceptedShare(CUTOFF, 2, 3)); // 49
export const SCANNED_4 = BINS_4 * 4 ** 3; // Å³ of bins a thread scans
export const SCANNED_2 = BINS_2 * 2 ** 3;

const n = (v: number) => v.toLocaleString('en-US');

export const questionsWrap: Record<string, Question> = {
  'ch21/wrap/q-sparse-bias': {
    kind: 'choice',
    id: 'ch21/wrap/q-sparse-bias',
    prompt: (
      <>
        The kernel above adds a list of {PAIRS} (token id, bias) pairs to {n(VOCAB)} logits before sampling. It is
        correct, needs no atomic and gives the same bits on every run. What, if anything, would you change?
      </>
    ),
    options: [
      {
        label: 'Nothing: one writer per output and no atomics is exactly the form this chapter recommends.',
        why: `It pays for every pair: ${n(VOCAB)} threads each test ${PAIRS} entries, ${n(VOCAB * PAIRS)} comparisons for ${PAIRS} useful additions. Gather won in this chapter because every atom contributes to every grid point.`,
      },
      {
        label: `Give each of the ${PAIRS} pairs a thread that adds its bias to its logit with an atomic: ${PAIRS} updates that rarely collide, against ${n(VOCAB * PAIRS)} comparisons. If the bits must not change between runs, sort the pairs by id and let one thread sum each id's run.`,
        correct: true,
      },
      {
        label: 'Make the add atomic: two threads race when a token appears in the list twice.',
        why: 'Only thread v ever writes logit v. A token that appears twice is summed inside that one thread.',
      },
      {
        label: 'Move ids and bias to shared memory: every thread reads them at a different address, so the loads are uncoalesced.',
        why: 'At every step all the lanes of a warp read the same ids[e] and bias[e]: one address for the whole warp. The store to logits is contiguous across the warp.',
      },
    ],
    explain: (
      <>
        <p>
          The chapter prefers gather because in direct summation every input reaches every output. The scatter kernel's
          lanes then collide on one point at every step, and gather does the same number of pairs without a single
          atomic. Here each input reaches one output. The gather kernel still visits every pair, with an equality test
          in place of 1/r, while a scatter does one update per input and its atomics almost never meet. Chapter 22 makes
          the same distinction: an input-centric thread may update one output, a subset or all of them.
        </p>
        <p>
          Sorting the pairs by id is binning with one bin per token: each thread then searches a short run instead of the
          whole list, and the sum's order is fixed. That is PyTorch's CUDA embedding gradient (sort, then one writer per
          row). PyTorch lists <code>index_add</code> and <code>scatter_add_</code> on CUDA among the operations that are
          nondeterministic unless you ask for deterministic algorithms. llama.cpp's <code>get_rows_back</code> is the
          gather form above, for training.
        </p>
      </>
    ),
    source:
      'ch. 21 §21.2, pp. 517–519; ch. 22 §22.2, p. 532 (notes/ch22.md); the kernel and the bias setting are ours; notes/today/ch21-facts.md D1, D3',
  },
  'ch21/wrap/q-intensity': {
    kind: 'numeric',
    id: 'ch21/wrap/q-intensity',
    prompt: (
      <>
        Count the gather kernel's arithmetic per atom per point from Fig. 21.6: {OPS_PER_ATOM} operations, square root
        not counted. With a chunk of {CHUNK} atoms, how many of those operations does one thread do per byte of
        global-memory traffic in one launch? Its only global access is the <code>+=</code> into its point.
      </>
    ),
    answer: INTENSITY,
    tol: 0.02,
    unit: 'operations per byte',
    explain: (
      <p>
        {CHUNK} × {OPS_PER_ATOM} = {n(CHUNK * OPS_PER_ATOM)} operations against {RMW_BYTES} bytes, one float read and one
        written: {INTENSITY} per byte. At a chunk of 1,024 it is {n(INTENSITY_1024)}. The atom reads go to the constant
        cache, which the book says removes most of their DRAM accesses. Chapter 5's example GPU needs 20 to leave the memory-bound regime (the book's
        example, p. 97), so the book is right that global bandwidth does not limit this kernel (p. 519). It also means the
        write pattern, the subject of §21.4, is a small share of the time, and a smaller one the larger the chunk: the
        basis of our prediction for lab 21.1.
      </p>
    ),
    source: 'counts from Fig. 21.6, p. 518 (notes/today/ch21-facts.md E1, E3); ch. 5 §5.1, p. 97; the reasoning is ours',
  },
  'ch21/wrap/q-factor8': {
    kind: 'numeric',
    id: 'ch21/wrap/q-factor8',
    prompt: (
      <>
        Coarsen by {FACTOR} with adjacent ownership, the thread index scaled: thread <code>t</code> owns points{' '}
        <code>{FACTOR}t</code> to <code>{FACTOR}t + {FACTOR - 1}</code> and writes them with {FACTOR} statements. On an
        NVIDIA warp of 32 lanes, with the row starting on a 128-byte boundary, how many 32-byte sectors do the{' '}
        {FACTOR} write statements touch together?
      </>
    ),
    answer: SECTORS_ADJ_ALL,
    unit: 'sectors',
    explain: (
      <p>
        In one statement lane <code>t</code> writes element <code>base + {FACTOR}t + m</code>: neighbouring lanes are{' '}
        {FACTOR} floats, 32 bytes, apart, so each lane has a sector to itself. {SECTORS_ADJ_STMT} sectors per statement,{' '}
        {SECTORS_ADJ_ALL} for the {FACTOR}. At a factor of 4 it was {SECTORS_ADJ_F4} per statement, two lanes to a sector.
        The waste grows with the factor until every lane needs its own sector, which is the most a 32-lane statement of
        floats can touch. With Fig. 21.10's stride every statement stays at {SECTORS_STR_STMT}, {SECTORS_STR_ALL} in all,
        whatever the factor.
      </p>
    ),
    source: 'ch. 21 §21.4, pp. 521–522; NVIDIA\'s 32-byte sectors (notes/today/ch06-facts.md A2; ch21-facts E2); factor 8 is ours',
  },
  'ch21/wrap/q-small-bins': {
    kind: 'choice',
    id: 'ch21/wrap/q-small-bins',
    prompt: (
      <>
        In the book's worked case, 4 Å bins and a 12 Å cutoff, each block's list has {n(BINS_4)} bins and about{' '}
        {SHARE_4} % of the atoms a thread tests pass. You halve the bin side to 2 Å, keeping the book's rule that a block's
        points are one bin. What happens?
      </>
    ),
    options: [
      {
        label: `About ${SHARE_2} % of the tested atoms now pass, so fewer tests are wasted; but the list grows to ${n(BINS_2)} bins, and each block is a 4 × 4 × 4 cube of 64 threads.`,
        correct: true,
      },
      {
        label: 'Nothing: the cutoff sphere is the same, so every thread tests the same atoms.',
        why: `A thread tests every atom in its list's bins, not in its sphere. Smaller bins follow the sphere more closely: ${n(SCANNED_2)} Å³ scanned instead of ${n(SCANNED_4)}.`,
      },
      {
        label: 'Fewer tested atoms pass, because more of the bins are only partly inside the cutoff.',
        why: `More bins are partial, but each is smaller, and the volume scanned falls from ${n(SCANNED_4)} to ${n(SCANNED_2)} Å³. The share that passes rises to about ${SHARE_2} %.`,
      },
      {
        label: 'The list gets shorter, because each bin is smaller and holds fewer atoms.',
        why: `It takes more small bins to cover the same super sphere: ${n(BINS_2)} against ${n(BINS_4)}.`,
      },
    ],
    explain: (
      <p>
        Computed with this chapter's cutoff model: the super-sphere radius is the cutoff plus half the bin diagonal, so
        smaller bins shrink the padding around the sphere and more of what is scanned is inside it. The price is
        elsewhere. Every list entry means a cooperative load and its barriers, and there are about four times as many. A block of
        64 threads is at the bottom of NVIDIA's suggested range. And a bin an eighth the volume holds an eighth as many
        atoms on average, so their counts vary more relative to that average, and a capacity that fits most bins pads
        more. Divergence falls and overhead rises; which wins is a measurement.
      </p>
    ),
    source:
      'ch. 21 §21.5, pp. 525–527; src/sim/cutoff.ts (notes/today/ch21-facts.md E4); block size advice ch21-facts A4; halving the bins is ours',
  },
  'ch21/wrap/q-bitwise': {
    kind: 'choice',
    id: 'ch21/wrap/q-bitwise',
    prompt: (
      <>
        Lab 21.1 compares your gather and coarsened kernels with each other within a tolerance, not bit for bit. Neither
        uses an atomic, and each gives the same bits on every run. Why can the two not be expected to agree exactly?
      </>
    ),
    options: [
      {
        label:
          'Their arithmetic differs: the coarsened kernel gets dx1 to dx3 by adding multiples of blockDim.x · gridspacing to dx0, where gather subtracts the atom\'s x from its own point\'s x. Each step rounds, so the distances, and the sums, can differ in the last bits.',
        correct: true,
      },
      {
        label: 'They add the atoms in different orders.',
        why: 'Both walk each chunk from n = 0 upwards, and both are launched once per chunk in the same order.',
      },
      {
        label: 'The coarsened kernel adds into the grid once per four points, gather once per point.',
        why: 'Both make one += per point per launch: the coarsened kernel has four write statements, one for each of its points.',
      },
      {
        label: 'Without atomics a GPU kernel can still give different bits on different runs.',
        why: 'A thread\'s own additions happen in program order. With no atomic and no other thread writing its point, the same build on the same device repeats its bits.',
      },
    ],
    explain: (
      <p>
        <code>(x − a) + 64·s</code> and <code>(x + 64·s) − a</code> are equal as real numbers and not always as floats,
        because each operation rounds (appendix A). On Metal, fast math adds its own looser division and square root. So
        two correct kernels agree to a tolerance, while a determinism test compares a kernel with itself. That is the
        line PyTorch draws: deterministic mode promises the same output for the same input on the same software and
        hardware, not the same output as another algorithm.
      </p>
    ),
    source:
      'Figs. 21.6 and 21.10, pp. 518, 522; appendix A §A.6; notes/today/ch10-facts.md A9, ch21-facts C3, D1; the lab\'s check is ours',
  },
};

// ---------------------------------------------------------------- this machine's numbers for the intensity question
/** The rows of lab 21.1 on one backend: the reader's own where they passed, else the reference's. */
function preferredRows(run: BackendRun | undefined) {
  const lab = 'ch21/lab1_dcs';
  const mine = labStatus(run, lab, 'work') === 'pass' ? benchRows(run, lab, 'work') : [];
  return mine.length ? { rows: mine, whose: 'your kernels' } : { rows: benchRows(run, lab, 'solution'), whose: 'the reference solution' };
}
const at = (rows: BenchRow[], name: string, chunk: number) =>
  num(rows.find((r) => r.name === name && num(r.nx) === 1024 && num(r.chunk) === chunk)?.seconds);

function OnThisMachine() {
  const { reports, loading } = useReports();
  const lines = realBackends(reports).flatMap((b) => {
    const { rows, whose } = preferredRows(reports[b]);
    const r = (chunk: number) => {
      const adj = at(rows, 'coarse_adjacent', chunk);
      const str = at(rows, 'coarse_strided', chunk);
      return adj && str ? adj / str : null;
    };
    const r64 = r(64);
    const r1024 = r(1024);
    if (r64 == null || r1024 == null) return [];
    return [{ b, device: String(reports[b].device.name ?? b), r64, r1024, whose }];
  });
  if (loading) return null;
  return lines.length ? (
    <>
      {lines.map((l) => (
        <p key={l.b} data-machine={l.b} data-r64={fmt(l.r64)} data-r1024={fmt(l.r1024)}>
          Your own numbers, {l.device} ({l.b}), {l.whose}: in lab 21.1 the strided kernel ran <b>{fmt(l.r64)}</b> times as
          fast as the adjacent one at chunk 64 and <b>{fmt(l.r1024)}</b> times at chunk 1,024. Above 1 the strided kernel
          was faster. The <a href={href('ch21', 'labs')}>labs page</a> has the full table.
        </p>
      ))}
    </>
  ) : (
    <p className="muted" data-machine="none">
      To put your own device beside this question you need lab 21.1's benchmark rows:{' '}
      <code>python3 labs/pmpp.py run 21.1 --bench</code>.
    </p>
  );
}

// ---------------------------------------------------------------- the page
export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = realBackends(reports);
  const lab = 'ch21/lab1_dcs';
  const passing = [...backends, 'emu'].filter((b) => labStatus(reports[b], lab, 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch21/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul>
        <li>
          Decide who owns each output before you write the kernel. A thread per input scatters: its outputs collide, every
          update needs an atomic, same-address atomics are serialised, and a float sum built from them changes its last
          bits from run to run. A thread per output gathers: one plain write, in a fixed order.
        </li>
        <li>
          Gather wins outright when every input reaches every output, as in an all-pairs sum. When each input reaches only
          a few outputs, gather pays for pairs that do not exist; then bin or sort the inputs first, or accept a scatter.
        </li>
        <li>
          The best sequential code can have the wrong loop order for a GPU. Parallelise the output-centric order, accept a
          little redundant arithmetic, then win the reuse back inside each thread.
        </li>
        <li>
          Coarsening: one thread computes several outputs, loads each shared operand once and computes shared terms once.
          It cannot remove what each output needs for itself (here a division and a square root per pair), and it costs
          registers.
        </li>
        <li>
          Which outputs a coarsened thread owns decides coalescing. Hand them out <code>blockDim.x</code> apart, or write
          them with one vector store; change the write offsets and each output's arithmetic together. Fig. 21.8 as printed
          gets the ownership wrong.
        </li>
        <li>
          A value every lane reads at once is the constant cache's best case on NVIDIA and AMD. Constant memory is small,
          so feed it in chunks and let each launch add its part.
        </li>
        <li>
          All-pairs work grows with the square of the problem. A cutoff with bins makes it linear: a neighbourhood list of
          offsets computed once per block on the host, bins staged in shared memory, a per-thread test that diverges,
          bins of one capacity padded with dummies, and a slower path for the overflow.
        </li>
        <li>
          Neither the book nor any vendor puts a number on these gains, and Apple documents no coalescing rule and no cost
          for atomics. Predict from the model, then measure.
        </li>
      </ul>
      <Source section="21.6" pages="527–528" />

      <h2>A mixed check</h2>
      <p>
        These draw on the whole chapter and on situations the pages did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly on the first recorded try.` : ''}
      </p>
      <Beyond>
        <p>
          The bias kernel, the factor of 8 and the halved bins are ours. The bias kernel is front-end-checked and has
          never been run.
        </p>
      </Beyond>
      <Code
        tabs={[
          {
            label: 'CUDA (not a lab file: for this question only)',
            code: BIAS,
            status: 'front-end-checked; never run',
            tiers: { logits: 'global', ids: 'global', bias: 'global', v: 'reg', add: 'reg', e: 'reg' },
          },
        ]}
      />
      <Ask q={q['ch21/wrap/q-sparse-bias']} label="Read the code" />
      <Ask q={q['ch21/wrap/q-intensity']} label="Work it out" />
      <OnThisMachine />
      <Ask q={q['ch21/wrap/q-factor8']} label="Work it out" />
      <Ask q={q['ch21/wrap/q-small-bins']} />
      <Ask q={q['ch21/wrap/q-bitwise']} />

      <h2>The book's exercises</h2>
      <p>
        The chapter ends with five exercises (p. 528). Each is worked somewhere in this course; the book prints no answers,
        so the answers here are ours.
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
              <td>Host code that configures the grid and launches the gather kernel</td>
              <td>
                <code>launchDcsGather</code> in <a href={href('ch21', 'labs')}>lab 21.1</a>, with the chunk loop of p. 517;
                the data flow on <a href={href('ch21', '21.2')}>21.2</a>
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td>Operations per iteration at a coarsening factor of eight</td>
              <td>
                a question on <a href={href('ch21', '21.2')}>21.2</a>, and the explorer's operation table at F = 8
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td>Two disadvantages of more work per thread</td>
              <td>
                a question on <a href={href('ch21', '21.2')}>21.2</a>; the register cost also on the{' '}
                <a href={href('ch21', 'start')}>start page</a>
              </td>
            </tr>
            <tr>
              <td>4</td>
              <td>Vector stores for the coarsened kernel</td>
              <td>
                a question on <a href={href('ch21', '21.2')}>21.2</a>, and "one float4 store" in its write-statement
                explorer; not part of the lab
              </td>
            </tr>
            <tr>
              <td>5</td>
              <td>How a neighbourhood bin causes control divergence</td>
              <td>
                a question on <a href={href('ch21', '21.5')}>21.5</a>, the explorer's mask panel and the binned-block
                animation
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Where the PDF looks wrong</h2>
      <p>
        The copy of the book this course was built from has a few apparent slips in chapter 21, and a few statements that
        current documentation does not support. They are listed so that you are not thrown by them when you read the book
        alongside.
      </p>
      <ul data-list="errata">
        <li>
          Fig. 21.8's line 05 adds an unscaled <code>threadIdx.x</code> while each thread writes <code>i</code> to{' '}
          <code>i+3</code>, so neighbouring threads overlap and most of each block's points are never written. The text on
          p. 521 describes threads four points apart, which needs the index scaled by the factor (
          <a href={href('ch21', '21.2')}>21.2</a>; lab 21.1's baseline is the corrected kernel).
        </li>
        <li>
          The operation counts on pp. 519–521 do not add up: 9 operations per point on p. 519 and 12 on p. 521; 12
          divisions for four points where the code has 4; 11 additions and 6 multiplications for Fig. 21.8 where the
          listing has 12 and 8; z left out of the eliminated loads; no square roots counted anywhere (
          <a href={href('ch21', '21.2')}>21.2</a> shows the counted numbers beside the book's).
        </li>
        <li>p. 521 says two problems cause the uncoalesced writes and describes only one.</li>
        <li>
          Fig. 21.9 has threads computing "up to 8 potentials, skipping by half-warps", where the code uses a factor of 4
          and a stride of <code>blockDim.x</code>. Its "Padding waste" is never discussed in the text (
          <a href={href('ch21', '21.2')}>21.2</a>).
        </li>
        <li>
          Fig. 21.13 labels its radius "cutoff distance + length of bin diagonal"; the text and Fig. 21.14 say half the
          diagonal. Fig. 21.14's "9 fully covered" holds at a single radius, about 2.12 bin widths; over the rest of the
          range where the list has 21 bins, only 5 are fully inside (<a href={href('ch21', '21.5')}>21.5</a>).
        </li>
        <li>
          The opening (p. 513) lists a "gather-to-scatter transformation"; the chapter goes from scatter to gather. The
          same list names "fast boundary condition checking", which no section develops (
          <a href={href('ch21', 'start')}>start page</a>).
        </li>
        <li>
          The slice is indexed three ways: <code>grid.x*grid.y*z</code> with a float z (Fig. 21.3),{' '}
          <code>(grid.x*grid.y*z) / gridspacing</code> (Figs. 21.4, 21.5) and <code>int k = z / gridspacing</code> (Fig.
          21.6 on). Lab 21.1 passes <code>k</code> as an integer.
        </li>
        <li>
          No kernel tests a bound, and Fig. 21.5 has no atom count, so the threads past the end of a partial chunk would add
          stale atoms. The snippets on <a href={href('ch21', '21.2')}>21.2</a> and lab 21.1 add both.
        </li>
        <li>
          Small slips: <code>atom[]</code> and <code>atoms[]</code> both on p. 519; the far atoms' "implicit method" (p.
          523) is never described (<a href={href('ch21', '21.5')}>21.5</a> names the one VMD uses).
        </li>
      </ul>
      <p>And where current vendor documentation says something else, or nothing:</p>
      <ul data-list="dated">
        <li>
          Fig. 21.9's "half-warps" is the coalescing unit of NVIDIA's 2007 guide. Today NVIDIA states coalescing per warp,
          in 32-byte sectors; do not learn a half-warp rule. Its "64-256 threads" per block still matches NVIDIA's and
          AMD's advice.
        </li>
        <li>
          Fig. 21.5's float <code>atomicAdd</code> is current, but NVIDIA now calls it a legacy atomic and recommends{' '}
          <code>cuda::atomic_ref</code>. On HIP its form depends on <code>-munsafe-fp-atomics</code>; on Metal the grid must
          be typed <code>atomic_float</code>.
        </li>
        <li>
          "Significantly reduces" (p. 518) and "faster" (p. 522) are not quantified by the book or by any vendor; lab 21.1
          measures both on your GPU.
        </li>
        <li>
          p. 521, the coarsened kernel's register count "stays within the allowed limit": no count, compiler or device is
          given, so it cannot be checked. <code>--resource-usage</code> shows yours on CUDA and HIP; Metal exposes no
          register count.
        </li>
        <li>
          The constant cache that makes the gather kernel's atom reads cheap (p. 519) is NVIDIA's. AMD's scalar cache
          serves only reads that are the same across the wavefront, and Apple documents no constant cache at all.
        </li>
      </ul>
      <Aside>
        <p>
          These are our readings of one PDF, checked against current vendor documentation on 2026-11-20
          (notes/today/ch21-facts.md). If your copy differs, trust your copy.
        </p>
      </Aside>

      <h2>What this chapter left out</h2>
      <p>
        The chapter is condensed in this course, so here is what is not covered and where to find it. The VMD background,
        ion placement and the physics of the map (§21.1, pp. 513–514) are one paragraph on the start page, and the
        rendered molecule of Fig. 21.1 (p. 514) is not shown. The two sequential C listings, Figs. 21.3 and 21.4 (pp.
        515–516), are described in prose, the loop interchange and the hoisted dz² and dy², not stepped through. The
        host's slice-and-chunk data flow (p. 517) gets two sentences on page 21.2 and lives in lab 21.1's launch functions.
      </p>
      <p>
        From §21.5, the Ångström-level 3D geometry of Fig. 21.13 and the construction from 512 circles (p. 525) are
        replaced by the 2D explorer and the computed 3D totals. The binned kernel, which the book gives only as an outline
        (p. 526), is explained and animated but not built in a lab. The sub-volumes that let the host's overflow pass
        overlap the device (p. 527) get one sentence. The vector-store revision of exercise 4 (p. 528) is discussed and
        questioned on page 21.2 but is not part of the lab. The summary (§21.6, pp. 527–528) is folded into this page, and
        the three references (p. 528) are not reproduced. Nothing later in the course depends on what was left out.
      </p>

      <h2>The project so far</h2>
      <p>
        This chapter adds no milestone to the through-line project. Its first decision is already in the milestones you
        have: the matrix multiplications of M2 and M3 give each thread its own outputs, the decomposition chapter 22 calls
        output-centric. Lab 21.1 is optional.{' '}
        <span data-lab211={passing.join(',') || 'none'}>
          {backends.length === 0 && !reports['emu']
            ? 'No lab results are recorded on this machine yet.'
            : passing.length
              ? `Your own version passes on ${passing.map(where).join(' and ')} on this machine.`
              : 'Your own version has not passed on this machine yet.'}
        </span>{' '}
        See the <a href={href('ch21', 'labs')}>lab</a>.
      </p>
      <p>
        Chapter 22 steps back from single kernels to the decisions behind them. It uses cutoff binning as its example of
        trading a little accuracy for a better algorithm (<a href={href('ch22', '22.1')}>§22.1</a>), sorts every pattern in
        the book into gather or scatter (<a href={href('ch22', '22.2')}>§22.2</a>), and ends with batching, latency
        against throughput, which is milestone M8. Start at <a href={href('ch22', 'start')}>chapter 22</a>.
      </p>
    </>
  );
}
