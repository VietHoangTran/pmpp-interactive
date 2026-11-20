import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { warpAccess } from '../../sim/coalesce';

// A kernel of our own, for two questions. Checked copy: labs/ch06/snippets/wrap_scores.cu
// (snippets-wrap.test.ts ties the two and pins the numbers below).
const SCORES = `// Attention scores for one query per head: s[h][t] = dot(q[h], K[t][h]).
// The key cache K is stored (token, head, dim), row-major. One thread per (head, token).
__global__ void scoresKernel(const float* q, const float* K, float* s, int T, int H, int D) {
    int h = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T) {
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) {
            acc += q[h * D + d] * K[(t * H + h) * D + d];
        }
        s[h * T + t] = acc;
    }
}`;

// ---------------------------------------------------------------- the numbers the questions take from the model
// NVIDIA's model (notes/today/ch06-facts.md A2): one warp's load costs the distinct
// 32-byte sectors its lanes touch. Byte addresses of one warp's 32 lanes, at one d.
const LANES = 32;
const SECTOR = 32;
export const SHAPE = { H: 32, D: 128, h: 5, d: 7 } as const;
const lanes = (f: (lane: number) => number) => Array.from({ length: LANES }, (_, l) => 4 * f(l));
const { H, D, h, d } = SHAPE;
/** The kernel as written: lane = token, so neighbouring lanes are H × D floats apart. */
export const SCORES_K = warpAccess(lanes((t) => (t * H + h) * D + d), 4, SECTOR);
/** The q load: every lane reads the same float. */
export const SCORES_Q = warpAccess(lanes(() => h * D + d), 4, SECTOR);
/** The same lanes reading a float4 of their own row instead (d a multiple of 4). */
export const SCORES_K4 = warpAccess(lanes((t) => (t * H + h) * D + 4), 16, SECTOR);
/** Remapped: one token per warp, lanes split d. */
export const SCORES_REMAPPED = warpAccess(lanes((l) => (3 * H + h) * D + l), 4, SECTOR);

/** Permuting (T, H, D) to (H, T, D): one warp of consecutive output elements, read and write side. */
export const PERMUTE = { T: 1000, H: 32, D: 128, first: 4096 * 37 + 64 } as const;
const outToIn = (o: number) => {
  const { T, H: nh, D: nd } = PERMUTE;
  const dd = o % nd;
  const t = Math.floor(o / nd) % T;
  const hh = Math.floor(o / (nd * T));
  return (t * nh + hh) * nd + dd;
};
export const PERMUTE_WRITE = warpAccess(lanes((l) => PERMUTE.first + l), 4, SECTOR);
export const PERMUTE_READ = warpAccess(lanes((l) => outToIn(PERMUTE.first + l)), 4, SECTOR);

/** Does every row of a K-float row-major matrix start on a 16-byte boundary (base aligned)? */
export const rowsAligned = (K: number) => (4 * K) % 16 === 0;
/** How many of the first `rows` rows start on a 16-byte boundary. */
export const alignedRows = (K: number, rows: number) => Array.from({ length: rows }, (_, r) => (4 * r * K) % 16 === 0).filter(Boolean).length;

export const questionsWrap: Record<string, Question> = {
  'ch06/wrap/q-scores-sectors': {
    kind: 'numeric',
    id: 'ch06/wrap/q-scores-sectors',
    prompt: (
      <>
        Take <code>scoresKernel</code> above with H = {H} heads and D = {D}. At one iteration <code>d</code>, how many
        32-byte sectors does one 32-lane warp&rsquo;s load from <code>K</code> touch, in NVIDIA&rsquo;s model?
      </>
    ),
    answer: SCORES_K.transactions,
    unit: 'sectors',
    explain: (
      <p>
        Neighbouring lanes are neighbouring tokens, and <code>t</code> is multiplied by H × D = {H * D} floats, so the
        lanes are {4 * H * D} bytes apart: each needs a sector of its own, {SCORES_K.transactions} in all, and uses 4 of
        its 32 bytes ({Math.round(SCORES_K.efficiency * 1000) / 10}%). The load from <code>q</code> is the opposite
        case: every lane reads the same float, {SCORES_Q.transactions} sector for the whole warp. Same rule as
        section 6.1&rsquo;s column-major N, in a kernel you will meet in attention.
      </p>
    ),
    source: 'rule of ch. 6 §6.1, pp. 126–129; sector model, notes/today/ch06-facts.md A2; the kernel is ours',
  },
  'ch06/wrap/q-scores-fix': {
    kind: 'choice',
    id: 'ch06/wrap/q-scores-fix',
    prompt: (
      <>
        You may not change how the key cache is stored: other kernels write it. Which change to{' '}
        <code>scoresKernel</code> makes its reads of <code>K</code> coalesce?
      </>
    ),
    options: [
      {
        label: 'Give each warp one token and let its 32 lanes split the d loop, then add their 32 partial sums.',
        correct: true,
      },
      {
        label: 'Read K with float4 loads, four values of d per lane.',
        why: `Each lane then uses 16 bytes of its sector instead of 4, but its neighbour is still ${4 * H * D} bytes away: ${SCORES_K4.transactions} sectors per warp, as before.`,
      },
      {
        label: 'Pad each row of K by one float.',
        why: 'Padding moves a column off one bank of shared memory. In global memory the lanes would still be H × D + 1 floats apart, each in its own sector.',
      },
      {
        label: 'Stage K in shared memory first, with the same thread mapping.',
        why: 'The load into shared memory would be the same scattered load. Corner turning works because the threads change which element they load, and that is a remapping.',
      },
    ],
    explain: (
      <p>
        This is the book&rsquo;s first remedy: change which thread handles which data. With lanes along <code>d</code>,
        32 lanes read 32 consecutive floats of one token&rsquo;s key, {SCORES_REMAPPED.transactions} sectors, all bytes
        used. The price is a sum across the lanes of a warp at the end, which chapter 10 does efficiently. The
        book&rsquo;s second remedy, a different layout such as (head, dim, token), would also work, at the cost of every
        kernel that writes the cache.
      </p>
    ),
    source: 'ch. 6 §6.1, p. 129 (three strategies) and Fig. 6.13 row 4, p. 148; the kernel is ours',
  },
  'ch06/wrap/q-permute': {
    kind: 'choice',
    id: 'ch06/wrap/q-permute',
    prompt: (
      <>
        Splitting heads: a kernel copies Q from shape (T, H, D) to (H, T, D), row-major floats, D = {PERMUTE.D}, one
        thread per element, with consecutive threads writing consecutive output elements. Which of its accesses
        coalesce?
      </>
    ),
    options: [
      { label: 'Both the reads and the writes.', correct: true },
      {
        label: 'Only the writes: the reads jump by H × D.',
        why: `Consecutive output elements differ first in d, and d is the innermost dimension of the input too. A warp reads ${PERMUTE_READ.transactions} sectors.`,
      },
      { label: 'Only the reads.', why: 'The writes are consecutive by construction: consecutive threads, consecutive output elements.' },
      {
        label: 'Neither: it is a transpose.',
        why: 'It swaps the two outer dimensions only. The case where one side turns strided is a transpose that moves the innermost dimension, like lab 6.1\'s.',
      },
    ],
    explain: (
      <p>
        Output element (h, t, d) comes from input element (t, h, d). For {PERMUTE.D} consecutive output elements h and t
        are fixed and d runs from 0 to {PERMUTE.D - 1}, so the input addresses are consecutive as well. A 32-lane warp
        writes {PERMUTE_WRITE.transactions} sectors and reads {PERMUTE_READ.transactions}. A permutation that keeps the
        innermost dimension in place copies contiguous runs of that dimension&rsquo;s length; only when the runs are
        shorter than a warp does a warp straddle several of them.
      </p>
    ),
    source: 'ch. 6 §6.1, pp. 126–128; data layout strategies, Fig. 6.13, p. 148; the example is ours',
  },
  'ch06/wrap/q-rowalign': {
    kind: 'choice',
    id: 'ch06/wrap/q-rowalign',
    prompt: (
      <>
        A matrix of K floats per row comes from <code>cudaMalloc</code>, and each thread reads its row through{' '}
        <code>float4</code> loads starting at the row&rsquo;s first element. For which K does every row start where a{' '}
        <code>float4</code> load is allowed?
      </>
    ),
    options: [
      { label: 'K a multiple of 4.', correct: true },
      { label: 'Any K: the allocation is aligned to 256 bytes.', why: 'Only the first row starts at the allocation. Row r starts 4·r·K bytes later.' },
      { label: 'K a power of two.', why: 'Sufficient from 4 upwards, but not necessary: K = 12 works too.' },
      { label: 'K a multiple of 32, the warp width.', why: 'Alignment is a property of each address, not of the warp. 16 bytes is 4 floats.' },
    ],
    explain: (
      <p>
        Row r starts 4·r·K bytes after the aligned base, and a <code>float4</code> access needs a 16-byte boundary, so 16
        must divide 4·K: K a multiple of 4. With K = 4097 only every fourth row starts on a 16-byte boundary
        ({alignedRows(4097, 8)} of the first 8); with K = 4100 every row does ({alignedRows(4100, 8)} of 8). Real code
        guards exactly this: llama.cpp&rsquo;s Metal matrix multiply falls back to element-wise reads when
        the row length breaks the vector type&rsquo;s alignment, and PyTorch picks its vector width from each
        pointer&rsquo;s alignment at run time.
      </p>
    ),
    source: 'notes/today/ch06-facts.md A3 (256-byte allocations), A11 (alignment to the vector size), D2 and D4',
  },
  'ch06/wrap/q-threebuf': {
    kind: 'choice',
    id: 'ch06/wrap/q-threebuf',
    prompt: (
      <>
        Lab 6.5 gives each operand two tiles and keeps one barrier per phase, after the loads. A colleague gives each
        operand three tiles, chosen by <code>ph % 3</code>, and removes that last barrier too. Correct?
      </>
    ),
    options: [
      {
        label: 'No: that barrier guards a true dependence. Threads read values other threads loaded this phase, and no number of buffers makes them arrive sooner.',
        correct: true,
      },
      {
        label: 'Yes: with three buffers no thread can overwrite a tile someone is still reading.',
        why: 'That was the false dependence, and two buffers already removed it. The remaining barrier is about reading a value before it is written.',
      },
      {
        label: 'Yes, on hardware that runs a warp in lockstep.',
        why: 'A 16 × 16 block is eight warps of 32, and nothing orders one warp\'s loads before another warp\'s reads.',
      },
      {
        label: 'No: three buffers need two barriers per phase.',
        why: 'More buffers never add barriers. They only make the write-after-read wait unnecessary.',
      },
    ],
    explain: (
      <p>
        Buffers separate locations, and that removes write-after-read waits only. The barrier after the loads is a
        read-after-write wait: the dot product reads tile slots that other threads write in the same phase. What more
        buffers can do is let the next phase&rsquo;s loads start early, which needs a copy the threads do not wait for;
        that is where chapter 15 takes double buffering.
      </p>
    ),
    source: 'ch. 6 §6.7, pp. 146–147 (true and false dependences); ch. 5 §5.4, pp. 110–111',
  },
  'ch06/wrap/q-bottleneck': {
    kind: 'choice',
    id: 'ch06/wrap/q-bottleneck',
    prompt: (
      <>
        A profiler reports two things about one kernel: its DRAM throughput is at 92% of the device&rsquo;s peak, and 40%
        of its shared-memory accesses have bank conflicts. What do you change first?
      </>
    ),
    options: [
      {
        label: 'Look for ways to move fewer bytes from DRAM (more reuse, smaller types), then profile again.',
        correct: true,
      },
      {
        label: 'Pad the shared tiles: 40% is a lot of conflicts.',
        why: 'Relieving a resource that is not the bottleneck buys nothing (p. 153). At 92% of its peak, DRAM is the one that is short.',
      },
      {
        label: 'Switch to vector loads, to move the same bytes in fewer instructions.',
        why: 'The bytes are the limit, not the instructions that move them, and there is little bandwidth left for faster issue to use.',
      },
      {
        label: 'Raise occupancy, to hide more latency.',
        why: 'More requests in flight help while bandwidth sits idle. Here it is nearly used up.',
      },
    ],
    explain: (
      <p>
        Near the memory roof the only lever left is fewer bytes. The bank conflicts are real, and once DRAM traffic falls
        they may become the bottleneck: that is why you profile again after each change and judge it by the
        kernel&rsquo;s duration, not by a percentage.
      </p>
    ),
    source: 'ch. 6 §6.9, pp. 153–154; Nsight Compute Compute Triage Guide, notes/today/ch06-facts.md A17; the numbers are made up',
  },
};

// ---------------------------------------------------------------- the page
export function Wrap() {
  const { progress } = useProgress();
  const { reports, loading } = useReports();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const m2 = backends.filter((b) => labStatus(reports[b], 'ch06/lab4_linear', 'work') === 'pass');
  const name = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch06/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <p>
        The chapter closes Part 1 of the book. With it you can reason about where any kernel&rsquo;s time goes, and
        these are the parts to keep.
      </p>
      <Source section="6.10" pages="154" />
      <ul>
        <li>
          A warp&rsquo;s load costs the distinct memory units its lanes touch, not the number of lanes. Lanes that read
          neighbouring addresses share units; multiply the thread index by anything larger than one and the count
          grows. The unit is the vendor&rsquo;s: 32-byte sectors on NVIDIA, 64- or 128-byte lines on AMD, none documented
          by Apple.
        </li>
        <li>
          When the natural mapping does not coalesce there are three ways out: change which thread takes which data,
          change the layout, or bring the data into shared memory in a coalesced order and do the awkward access there
          (corner turning). A linear layer with its weight stored (out, in) is that case.
        </li>
        <li>
          DRAM gives its bandwidth only to many requests in flight, spread over channels and banks. Resident warps
          supply them, and so do wider requests: occupancy and vector loads are bandwidth tools.
        </li>
        <li>
          Shared memory is split into banks, and lanes that land in the same bank are served one after another. A stride
          that shares a factor with the bank count is the usual culprit; one column of padding moves each row by one
          bank. The bank counts are NVIDIA&rsquo;s and AMD&rsquo;s, not Apple&rsquo;s.
        </li>
        <li>
          Coarsening, unrolling and double buffering each spend something (parallelism, registers, shared memory) to
          remove something (overhead, stalls, a barrier). Each pays only when what it removes was costing time.
        </li>
        <li>
          Find the bottleneck, relieve it, measure, repeat, and stop near the roof. The kernel&rsquo;s duration is the
          ground truth, and the device decides: the same rule can be worth a factor on one GPU and nothing on another.
        </li>
      </ul>

      <h2>A mixed check</h2>
      <p>
        These use the whole chapter on situations the sections did not show, most of them from inference kernels.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly on the first recorded try.` : ''}
      </p>
      <Beyond>
        <p>
          The kernels and shapes in these questions are ours. The key-cache layout in the first is one possible choice,
          picked for the question.
        </p>
      </Beyond>
      <Code
        tabs={[
          {
            label: 'CUDA (not a lab file: for these questions only)',
            code: SCORES,
            status: 'front-end-checked; never run',
            tiers: { q: 'global', K: 'global', s: 'global', acc: 'reg', t: 'reg', d: 'reg' },
          },
        ]}
        caption={<>From <code>labs/ch06/snippets/wrap_scores.cu</code>.</>}
      />
      <Ask q={q['ch06/wrap/q-scores-sectors']} />
      <Ask q={q['ch06/wrap/q-scores-fix']} />
      <Ask q={q['ch06/wrap/q-permute']} />
      <Ask q={q['ch06/wrap/q-rowalign']} />
      <Ask q={q['ch06/wrap/q-threebuf']} />
      <Ask q={q['ch06/wrap/q-bottleneck']} />

      <h2>The book&rsquo;s exercises</h2>
      <p>
        The chapter ends with five exercises (pp. 154–155). Each is worked somewhere in this course; doing the
        book&rsquo;s own wording as well is worth it.
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
              <td>Classify nine accesses of a kernel as coalesced, uncoalesced or not applicable</td>
              <td>
                the exercise 1 drill and three review questions in <a href={href('ch06', '6.1')}>6.1</a>
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td>A matrix multiplication with corner turning</td>
              <td>
                the corner-turning stepper in <a href={href('ch06', '6.1')}>6.1</a>, then lab 6.4 on the{' '}
                <a href={href('ch06', 'labs')}>labs page</a>, in the (out, in) weight layout, which is the same access
                problem
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td>Which tile widths avoid uncoalesced accesses in chapter 5&rsquo;s kernel</td>
              <td>
                the question and the stepper&rsquo;s T = 8, 16, 32 in <a href={href('ch06', '6.1')}>6.1</a>, under both
                readings of the exercise
              </td>
            </tr>
            <tr>
              <td>4</td>
              <td>Vector-load vector addition with its boundaries</td>
              <td>
                the chunk planner in <a href={href('ch06', '6.3')}>6.3</a>, then lab 6.2 on the{' '}
                <a href={href('ch06', 'labs')}>labs page</a>
              </td>
            </tr>
            <tr>
              <td>5</td>
              <td>Banks and conflicts for seven strides</td>
              <td>
                the exercise 5 table and the bank explorer in <a href={href('ch06', '6.4')}>6.4</a>; lab 6.3 times the
                same strides
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Where the PDF looks wrong</h2>
      <p>
        The copy this course was built from has a few apparent slips and loose ends in chapter 6. They are listed so
        that you are not thrown by them when you read the book alongside.
      </p>
      <Aside>
        <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
      </Aside>
      <ul data-list="errata">
        <li>
          pp. 140–141 print the shift the wrong way round: multiplying by 32 is <code>i &lt;&lt; 5</code>, and by 33 is{' '}
          <code>(i &lt;&lt; 5) + i</code>. As printed, <code>i &gt;&gt; 5 + i</code> is a right shift by 5 + i bits. See{' '}
          <a href={href('ch06', '6.4')}>6.4</a>.
        </li>
        <li>
          p. 147 says double buffering eliminates the &ldquo;second&rdquo; barrier; the one it removes, by its own
          analysis and code, is the first, after the read. In lab 6.5&rsquo;s loop, which loads first, it is the second.
          See <a href={href('ch06', '6.7')}>6.7</a>.
        </li>
        <li>
          p. 133 gives the HBM3E bus clock as &ldquo;4.9 GHz/s&rdquo; (GHz is meant) and the result as 1 TB/s. Its own
          inputs give 8 bytes × 2 × 4.9 GHz × 16 = 1254.4 GB/s, about 1.25 TB/s, for one stack. See{' '}
          <a href={href('ch06', '6.2')}>6.2</a>.
        </li>
        <li>
          p. 146 says local arrays live in &ldquo;global memory&rdquo; by default; chapter 5 calls the place local
          memory, and so does NVIDIA, which puts it physically in device memory and lists when an array goes there.
          See <a href={href('ch06', '6.6')}>6.6</a>.
        </li>
        <li>
          Fig. 6.12 has no bounds checks and never uses its parameter <code>N</code>; the text leaves the boundaries to
          Exercise 4, so the listing is not complete as printed. Its caption says &ldquo;vector loads&rdquo;, though
          line 10 is a vector store as well.
        </li>
        <li>
          p. 137 asks you to check that 8 × 8 matrices &ldquo;use all four channels&rdquo;. With Fig. 6.9&rsquo;s
          mapping they do across the run, but each phase&rsquo;s M loads use one channel, fewer at once than the 4 × 4
          case. The book does not say which it means.
        </li>
        <li>
          Fig. 6.8&rsquo;s legend calls the light segments &ldquo;idle time&rdquo; while the text calls them access
          latency, and it draws a ratio of about 2 to 1 where the text works with 20 to 1.
        </li>
        <li>
          The names drift: Fig. 6.4 calls the shared tiles <code>A_s</code> and <code>B_s</code> for matrices M and N,
          §6.4 uses <code>a</code> and <code>TILE_DIM</code>, and Exercise 3 asks about a <code>BLOCK_SIZE</code> that
          chapter 5&rsquo;s kernel does not have (it uses <code>TILE_WIDTH</code>).
        </li>
        <li>
          Exercise 3 has two defensible answers. If &ldquo;coalesced&rdquo; means one run of 32 addresses per warp,
          only a tile width of 32 qualifies; under NVIDIA&rsquo;s sector rule an aligned 16-wide tile wastes nothing
          either. Exercise 5&rsquo;s &ldquo;number of bank conflicts&rdquo; could mean lanes per bank or extra passes;
          this course gives lanes per bank, the book&rsquo;s &ldquo;k-way&rdquo; usage.
        </li>
        <li>
          The checklist (Fig. 6.13) and its text differ in two rows: register tiling&rsquo;s table entry speaks of shared
          memory only, the text of global or shared memory; and warp-level primitives&rsquo; table entry claims less
          global traffic, which the text does not explain.
        </li>
        <li>
          The chapter never gives a burst or cache-line size for a real GPU, nor what happens to partly consecutive
          accesses such as a stride of 2. Section <a href={href('ch06', '6.1')}>6.1</a> fills that in from NVIDIA&rsquo;s
          and AMD&rsquo;s documents.
        </li>
        <li>
          The summary (p. 154) leaves out vector loads and bank conflicts, although sections 6.3 and 6.4 cover them.
        </li>
        <li>
          Small typos: the sidebar is cited as &ldquo;Why is DRAM so slow?&rdquo; but titled &ldquo;Why are DRAMs so
          slow?&rdquo; (p. 124); &ldquo;the accesses data wall span&rdquo; (p. 126); &ldquo;Bank 0 or Channel 1&rdquo;
          for &ldquo;of&rdquo; (p. 135); &ldquo;integer of floating point&rdquo; for &ldquo;or&rdquo; (p. 139).
        </li>
      </ul>
      <p>
        One of the book&rsquo;s dated statements is contradicted today: &ldquo;one to eight channels&rdquo; (p. 133)
        understates current GPUs. Two could not be confirmed either way: NVIDIA gives no shared-memory latency in
        cycles, for &ldquo;a few clock cycles&rdquo; (p. 139), and no vendor page we read gives DRAM latency in
        nanoseconds, for &ldquo;tens of nanoseconds&rdquo; (p. 124). Both figures are the book&rsquo;s.
      </p>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md A1, A7, A12)">
        <p>
          NVIDIA gives global-memory latency as &ldquo;hundreds of clock cycles&rdquo;. AMD calls 8 to 16
          independent memory controllers typical.
        </p>
      </Today>

      <h2>What this course shortened</h2>
      <p>
        Every section of the chapter has its own page here. Three passages are shorter than in the book: the sidebar on
        why DRAM is slow (p. 125) is one paragraph without its circuit figure, the carpool photographs and traffic
        policies (Figs. 6.5 and 6.6, pp. 131–132) are one sentence, and the per-chapter forward references of §6.8
        (pp. 149–153) are the checklist tool&rsquo;s &ldquo;returns in&rdquo; column. The one reference (p. 155) is
        replaced by the dated notes beside each section. Asynchronous copies, loads wider than 16 bytes and profiler
        walkthroughs are only mentioned; chapter 15 is where they belong.
      </p>

      <h2>The project so far</h2>
      <p>
        Lab 6.4 is milestone M2 &ldquo;measured and tuned&rdquo;: the project&rsquo;s linear layer, now in the weight
        layout PyTorch and llama.cpp use, with a corner-turned, padded weight tile and a benchmark against the
        uncoalesced version.{' '}
        <span data-m2={loading ? 'loading' : m2.join(',') || 'none'}>
          {loading
            ? 'Reading reports.'
            : backends.length === 0
              ? 'No lab results are recorded on this machine yet.'
              : m2.length
                ? <>Your own <code>linearKernel</code> passes on {m2.map(name).join(' and ')} on this machine.</>
                : <>Your own <code>linearKernel</code> has not passed on this machine yet.</>}
        </span>{' '}
        See the <a href={href('ch06', 'labs')}>labs</a>. Chapter 15 coarsens and register-tiles the same kernel, with the
        same signature, into milestone M3.
      </p>
      <p>
        Chapter 7 opens Part 2 of the book, the parallel patterns, with convolution. Every output there needs a
        neighbourhood of inputs, so neighbouring blocks need overlapping input tiles, and the tiles you load are larger
        than the tiles you produce. It also adds constant memory, and it judges each of its four kernels by operations
        per byte, the method of chapter 5. Start at <a href={href('ch07', 'start')}>chapter 7</a>.
      </p>
      <Source section="6.8" pages="151, 153" figs="the checklist's forward references to chapter 7" />
    </>
  );
}
