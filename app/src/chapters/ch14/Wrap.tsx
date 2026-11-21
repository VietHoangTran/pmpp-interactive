import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Beyond, Source, type Tier } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { functionSource } from '../../sim/toolchain';

// The kernel of the first question is a snippet of our own, loaded at run time so there is no
// copy to drift: labs/ch14/snippets/wrap_blocksort.cu. Wrap.snippets.test.ts runs it, and the
// corrected copy in labs/ch14/snippets/check/wrap_check.cu, on the course's emulator, and pins
// what the question says about both.
export const BLOCKSORT_FILE = 'ch14/snippets/wrap_blocksort.cu';
const TIERS: Record<string, Tier> = { keys: 'global', s: 'shared', t: 'reg', i: 'reg', step: 'reg', tmp: 'reg', n: 'reg' };

/** MSD select: keys per top-8-bit bin, from the top bin down, and the k of q-select-bin. */
export const SELECT_BINS = [0, 3, 12, 40, 97];
export const SELECT_K = 50;
/** The bin holding the k-th largest key, the keys above it, and how many it must still give. */
export function selectBin(bins: number[], k: number): { bin: number; above: number; still: number } {
  let above = 0;
  for (let b = 0; b < bins.length; ++b) {
    if (above + bins[b] >= k) return { bin: b, above, still: k - above };
    above += bins[b];
  }
  return { bin: -1, above, still: k - above };
}
/** Passes of an LSD radix sort with d-bit digits over the low `bits` bits of a key. */
export const passes = (bits: number, d: number) => Math.ceil(bits / d);
/** Bits of row id that `rows` rows need. */
export const rowBits = (rows: number) => Math.ceil(Math.log2(rows));

const SEL = selectBin(SELECT_BINS, SELECT_K);
const ROWS = 64;
const PACKED = passes(32 + rowBits(ROWS), 8);

export const questionsWrap: Record<string, Question> = {
  'ch14/wrap/q-blocksort': {
    kind: 'choice',
    id: 'ch14/wrap/q-blocksort',
    prompt: (
      <>
        The kernel above is launched as one block of <code>(n + 1) / 2</code> threads. A test on a GPU sorts 64 random keys
        with it and passes. What is wrong with it?
      </>
    ),
    options: [
      {
        label: 'Nothing: within a step the pairs are disjoint, so no two threads touch the same element, and the GPU test confirms it.',
        why: 'Within one step, yes. But element 2t + 2 belongs to thread t + 1 on an even step and to thread t on the odd step after it, and nothing makes the even step finish first.',
      },
      {
        label: 'A thread can start a step while its neighbour is still on the last one; one __syncthreads() per step fixes it.',
        correct: true,
      },
      {
        label: 'The swaps must be atomic, because two threads of one step can still swap the same element.',
        why: 'No two threads write the same element in the same step. What is missing is an order between steps, which a barrier gives and an atomic does not.',
      },
      {
        label: 'n steps are too few for a block-level sort; it needs section 14.2\'s flag and a loop that runs until nothing changes.',
        why: 'A fixed number of steps is what makes this a sorting network (section 14.9). The corrected kernel sorts every case the test gives it in n steps, including all 256 lists of eight 0s and 1s.',
      },
    ],
    explain: (
      <>
        <p>
          Every step's pairs overlap the previous step's. Without a barrier the language promises nothing about how far
          apart two threads of a block are, so thread t can compare a pair whose second element thread t + 1 has not yet
          swapped, or both can swap element 2t + 2 at the same moment and lose a value: section 14.2's neighbouring-pairs
          race, moved from inside one step to between two. A block of 32 threads is one warp on NVIDIA, where the threads
          may well keep in step; the language does not promise it, and a test that passes there proves nothing.
        </p>
        <p>
          On the course's emulator, which runs one thread at a time between barriers, the kernel as shown leaves 64, 129 and
          256 keys unsorted in all three thread orders. With <code>__syncthreads()</code> as the last statement of the loop
          body it sorts them all. That is the cost of a network in shared memory: a barrier per stage.
        </p>
      </>
    ),
    source: 'our kernel, after §14.2, pp. 331–333, and §14.9, p. 345; barriers: ch. 5 §5.4, pp. 110–111',
  },
  'ch14/wrap/q-atomic-scatter': {
    kind: 'choice',
    id: 'ch14/wrap/q-atomic-scatter',
    prompt: (
      <>
        To skip the scan in a one-bit radix pass, someone keeps two counters in global memory, one for the zero bucket
        starting at 0 and one for the one bucket starting at the number of zeros, and lets each thread take its key's
        destination with <code>atomicAdd</code> on its bucket's counter. After all 32 passes, what does the output hold?
      </>
    ),
    options: [
      {
        label: 'The keys sorted by value, but equal keys in no promised order: the atomics cost only stability.',
        why: 'That describes one pass. An LSD sort needs every pass to keep the previous order inside each bucket, and without that the later passes undo the earlier ones.',
      },
      {
        label: 'Every key once, but in general not sorted: atomics fill each bucket in arrival order, discarding the earlier passes\' work.',
        correct: true,
      },
      {
        label: 'Keys lost or duplicated: two threads of one bucket can read the same counter value before either adds, and both write that slot.',
        why: 'atomicAdd returns a different old value to every thread, so every slot is taken exactly once.',
      },
      {
        label: 'The keys sorted, as with the scan, only slower: each bucket\'s counter serialises its threads.',
        why: 'The scan gives each key the slot that keeps input order; the atomic gives it a slot in arrival order. Same bucket, different place in it.',
      },
    ],
    explain: (
      <p>
        After the last pass the keys are split correctly by bit 31, and that is all that is guaranteed. Inside each half
        the order should be the one the first 31 passes built, and the atomics replaced it with arrival order. Stability is
        not a nicety in an LSD radix sort: it is what carries each pass's work into the next (section 14.4). The output is
        still a permutation, which is why a test that checks only "every key is there" would pass it.
      </p>
    ),
    source: 'ch. 14 §14.4, pp. 335–336, and §14.5, pp. 337–338; atomics: ch. 9',
  },
  'ch14/wrap/q-packed-rows': {
    kind: 'numeric',
    id: 'ch14/wrap/q-packed-rows',
    prompt: (
      <>
        Sort every row of a batch of {ROWS} rows with one ordinary LSD radix sort: give each element a 64-bit key whose low
        32 bits are its probability's descending key and whose next bits hold its row id. The sort uses 8-bit digits and
        skips the high bits no key uses. How many passes does it make?
      </>
    ),
    answer: PACKED,
    unit: 'passes',
    explain: (
      <p>
        {ROWS} rows need {rowBits(ROWS)} bits of row id, so the keys use 32 + {rowBits(ROWS)} = {32 + rowBits(ROWS)} bits,
        and 8-bit digits need ceil({32 + rowBits(ROWS)} / 8) = {PACKED} passes. LSD order does the rest: the first four
        passes sort by probability, the last by row, and stability carries the probability order into each row. That is
        section 14.1's cascade, secondary key first and primary key last, inside one sort. Sorting each row on its own
        takes 4 passes; packing costs a pass and keys twice as wide, and saves a segmented sort. CUB's radix sort takes{' '}
        <code>begin_bit</code> and <code>end_bit</code>, so a packed 64-bit key like this one can be sorted on its low{' '}
        {32 + rowBits(ROWS)} bits only (A2).
      </p>
    ),
    source: 'our arithmetic, after §14.1, p. 330, and §14.7, pp. 342–344; CUB bit range: notes/today/ch14-facts.md A2',
  },
  'ch14/wrap/q-select-bin': {
    kind: 'numeric',
    id: 'ch14/wrap/q-select-bin',
    prompt: (
      <>
        A radix select looks for the {SELECT_K} largest keys of a row. Its histogram of the top 8 bits, read from the top
        bin down, holds {SELECT_BINS.join(', ')}, ... keys. How many keys must it still pick from inside the bin where the{' '}
        {SELECT_K}th largest key lies?
      </>
    ),
    answer: SEL.still,
    unit: 'keys',
    explain: (
      <p>
        Running totals from the top: {SELECT_BINS.slice(0, SEL.bin + 1).map((_, b) => SELECT_BINS.slice(0, b + 1).reduce((x, y) => x + y, 0)).join(', ')}. The{' '}
        {SELECT_K}th largest key is in bin {SEL.bin + 1} from the top. Every key of the bins above it is in the answer,{' '}
        {SEL.above} of them, and no key below it is, so the select keeps those without ever ordering them and goes on to the
        next digit only inside that bin, for the {SEL.still} still missing: {SELECT_BINS[SEL.bin]} keys looked at again
        instead of the whole row. That is section 14.9's most-significant-digit idea applied to selection; llama.cpp's{' '}
        <code>top_k_radix_cuda</code> walks 8-bit bins from the top the same way.
      </p>
    ),
    source: 'our arithmetic, after §14.9, p. 346; llama.cpp: notes/today/ch14-facts.md D2',
  },
  'ch14/wrap/q-topp-head': {
    kind: 'choice',
    id: 'ch14/wrap/q-topp-head',
    prompt: (
      <>
        A top-p sampler with p = 0.9 must cut every row exactly where a full sort would. On the row in front of you, the
        200 most likely tokens already hold 0.95 of the probability. What is the least ordering work the sampler can
        always rely on?
      </>
    ),
    options: [
      {
        label: 'A full sort of the row: the cut is defined on the sorted order, so every token\'s rank must be known.',
        why: 'Only the head of that order matters. Tokens past the cut are dropped whatever their order.',
      },
      {
        label: 'Select a set sure to contain the cut, here the 200 largest, and sort only those; widen it if their sum falls short of p.',
        correct: true,
      },
      {
        label: 'None: add up probabilities in token order until the running sum reaches 0.9, then cut there.',
        why: 'That keeps the first tokens by id, not the most likely ones.',
      },
      {
        label: 'Sort the 200 largest on every row and skip the check: this row shows that 200 tokens hold more than p.',
        why: 'A flat row can spread 0.9 over thousands of tokens; only the running sum shows whether the head was long enough.',
      },
    ],
    explain: (
      <p>
        The cut is the shortest prefix of the descending order whose sum reaches p, so only that prefix needs an order, and
        only the knowledge that nothing outside the candidates belongs in it. llama.cpp's host top-p does exactly this: with
        more than 1,024 candidates it partially sorts the top 256, takes the running sum, and widens the sort only if p was
        not reached. A GPU sampler can do the same with a radix select and a sort of the candidates.
      </p>
    ),
    source: 'our reasoning; llama.cpp: notes/today/ch14-facts.md D3',
  },
};

// ---------------------------------------------------------------- milestone M5 so far
const M5: { lab: string; num: string; ch: string; what: string }[] = [
  { lab: 'ch09/lab3_tokencounts', num: '9.3', ch: 'ch09', what: 'token counts for the repetition penalty' },
  { lab: 'ch11/lab6_topp', num: '11.6', ch: 'ch11', what: 'top-p keep mask over a sorted row' },
  { lab: 'ch12/lab4_candidates', num: '12.4', ch: 'ch12', what: 'candidate compaction' },
  { lab: 'ch14/lab3_sortrows', num: '14.3', ch: 'ch14', what: 'sorted candidates, every row descending and stable' },
];
const where = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
/** Backends (devices first, then the emulator) on which the reader's own work of a lab passes. */
export const passingOn = (reports: Reports, lab: string) =>
  [...realBackends(reports), 'emu'].filter((b) => labStatus(reports[b], lab, 'work') === 'pass');

export function Wrap() {
  const { progress } = useProgress();
  const { reports, loading } = useReports();
  const snip = useLabFile(BLOCKSORT_FILE);
  const kernel = snip && !snip.startsWith('// could not load') ? functionSource(snip, 'blockOddEvenSort')?.text ?? null : null;
  const ran = Object.keys(reports).length > 0;
  const own143 = passingOn(reports, 'ch14/lab3_sortrows');
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch14/') && !k.includes('/p-'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul>
        <li>
          A sort owes you two things: an order and a permutation. A stable sort also keeps equal keys in input order, which
          is what makes a sort by two keys work and what fixes a sampler's order among tied probabilities.
        </li>
        <li>
          Odd-even sort orders disjoint pairs, alternating even and odd: O(N) steps and O(N²) work, one launch per step and a
          flag the host must read. As a fixed loop inside a thread or a block it is the simplest sorting network, and in a
          block it needs a barrier per step.
        </li>
        <li>
          Merge sort sorts segments, then merges them in a tree. Early stages run many merges side by side; late stages split
          a few large merges between blocks with chapter 13's co-rank search. It is stable if every merge takes from the left
          run on ties.
        </li>
        <li>
          A radix pass is a stable partition by one digit. Each key's place comes from an exclusive scan of the bits, with
          the total at index n; every pass must be stable for the passes to add up; floats sort after a bit transform, and
          descending order comes from inverting the key, not reversing the result.
        </li>
        <li>
          The book's scan in the middle of a kernel has no portable form. Cut each pass into launches (extract or count,
          scan, scatter); Metal has no other way, and CUDA and HIP only under a cooperative launch.
        </li>
        <li>
          A direct scatter writes up to 2^r runs per warp. Sorting each block's keys locally in shared memory first, with a
          digit-major table of bucket sizes scanned for their starts, makes the writes coalesce. Wider digits mean fewer passes
          but more runs and a bigger table; more keys per thread mean longer runs and a smaller table, until shared memory
          runs out.
        </li>
        <li>
          Where a library sort exists, use it: CUB documents stable, descending, segmented radix sorts, and hipCUB has the
          same calls over rocPRIM, whose radix sort is documented as stable. Metal has no stable sort at kernel level. Top-k
          is a selection, not a sort.
        </li>
      </ul>
      <Source section="14.10" pages="346–347" />

      <h2>A mixed check</h2>
      <p>
        These draw on the whole chapter and on situations the sections did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
      </p>
      {kernel ? (
        <Code
          tabs={[
            {
              label: 'CUDA (not a lab file: for this question only)',
              code: kernel,
              tiers: TIERS,
              status: 'front-end-checked; emulator-checked; unverified on CUDA and HIP hardware',
            },
          ]}
          caption={
            <>
              From <code>labs/{BLOCKSORT_FILE}</code>. One block sorts up to 1,024 keys in shared memory with n fixed
              odd-even steps: no flag and no host loop.
            </>
          }
        />
      ) : null}
      <Ask q={questionsWrap['ch14/wrap/q-blocksort']} label="Find the defect" />
      <Ask q={questionsWrap['ch14/wrap/q-atomic-scatter']} />
      <Ask q={questionsWrap['ch14/wrap/q-packed-rows']} label="Work it out" />
      <Ask q={questionsWrap['ch14/wrap/q-select-bin']} label="Work it out" />
      <Ask q={questionsWrap['ch14/wrap/q-topp-head']} />

      <h2>The book's exercises</h2>
      <p>
        The chapter ends with four coding exercises (p. 347). The first three are one lab here; the fourth is sketched and
        rests on chapter 13's merge.
      </p>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="exercises">
        <thead>
          <tr><th>Exercise</th><th>About</th><th>Where it is worked here</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>1</td>
            <td>Fig. 14.7 with shared memory, for coalesced writes</td>
            <td>
              the warp explorer and bucket table in <a href={href('ch14', '14.6')}>14.6</a>; lab 14.2 at r = 1 on the{' '}
              <a href={href('ch14', 'labs')}>labs page</a>
            </td>
          </tr>
          <tr>
            <td>2</td>
            <td>A multi-bit radix</td>
            <td>
              the radix-width explorer in <a href={href('ch14', '14.7')}>14.7</a>; lab 14.2, r from 1 to 8
            </td>
          </tr>
          <tr>
            <td>3</td>
            <td>Thread coarsening</td>
            <td>
              Fig. 14.13 and the keys-per-thread explorer in <a href={href('ch14', '14.8')}>14.8</a>; lab 14.2, c from 1 to 8
            </td>
          </tr>
          <tr>
            <td>4</td>
            <td>Merge sort with chapter 13's parallel merge</td>
            <td>
              sketched in "Exercise 4, in outline" on <a href={href('ch14', '14.3')}>14.3</a>, not built; the merge itself is
              lab 13.1 on <a href={href('ch13', 'labs')}>chapter 13's labs page</a>
            </td>
          </tr>
        </tbody>
      </table>

      <h2>Where the PDF looks wrong</h2>
      <p>
        The copy this course was built from has a few apparent slips in chapter 14. The first five are on the page; the
        last four are our reading of places the page leaves unclear.
      </p>
      <ul data-list="errata">
        <li>p. 329 says the stable filter "will be discussed further in Chapter 12", which comes before this chapter. p. 336 has it right.</li>
        <li>
          p. 329 says "transposition sort and sampling sort are also discussed" as other methods. Section 14.9 covers sorting
          networks, sample sort and MSD radix sort; odd-even transposition sort is section 14.2's own algorithm.
        </li>
        <li>
          p. 337 introduces the second destination formula "for keys mapping to the zero bucket". The formula is for the one
          bucket, as the figure labels it (<a href={href('ch14', '14.5')}>14.5</a>).
        </li>
        <li>
          Fig. 14.7 reads <code>bits[N]</code> as the number of ones in total, which needs an exclusive scan of N + 1 entries;
          Fig. 14.6 shows N, and its last value equals the total only because the last key's bit is 0. Lab 14.1 scans n + 1
          entries.
        </li>
        <li>p. 339: "indeed to right destination indices" should read "to the right destination indices".</li>
        <li>
          Our reading: <code>gridExclusiveScan</code> in Fig. 14.7 is pseudocode. It is never defined, and every block must
          have written its bits before it and must wait for its end, which an ordinary launch cannot arrange. Section 14.5 and
          the labs split the pass into launches.
        </li>
        <li>
          Our reading: the odd-even host loop is not shown, and "until no elements are swapped" read as "until one step makes
          no swap" stops <code>[1, 3, 2, 4]</code> unsorted. Two steps in a row, one even and one odd in either order,
          both quiet, show that the list is sorted (<a href={href('ch14', '14.2')}>14.2</a>).
        </li>
        <li>
          Our reading: Fig. 14.13 draws each block's threads over half of its keys and does not say which thread owns which;
          eight keys per four-thread block means two per thread (<a href={href('ch14', '14.8')}>14.8</a>).
        </li>
        <li>
          Our reading: p. 334 gives each merge stage O(log N) time and O(N log N) work "as seen in Chapter 13", but chapter 13
          states O(m + n) for the sequential merge and about log2 N steps for one co-rank search, never a bound for a whole
          parallel merge (<a href={href('ch14', '14.3')}>14.3</a>).
        </li>
      </ul>
      <p>Two things the listings leave to you, which the lessons and labs correct and test:</p>
      <ul data-list="gaps">
        <li>
          Fig. 14.2 guards with <code>i &lt; N - 1</code> on an unsigned N, which wraps when N is 0; <code>i + 1 &lt; N</code>{' '}
          cannot.
        </li>
        <li>
          Fig. 14.2 sets <code>hasChanged</code> with a plain store from many threads. The book itself warns that this is a
          data race and undefined behaviour in the C++ memory model; the lessons and labs use an atomic store.
        </li>
      </ul>
      <Aside>
        <p>
          These are our readings of one PDF, from notes/ch14.md, "Unclear in the PDF". If your copy differs, trust your copy.
        </p>
      </Aside>

      <h2>What this chapter left out</h2>
      <p>
        Chapter 14 is a Standard chapter here: every section has a page, and these parts of the book were kept short or left
        out.
      </p>
      <ul data-list="left-out">
        <li>Cole's merge sort, O(log N) time and O(N log N) work (p. 334), is named only, by the book and by us.</li>
        <li>
          Parallel merge sort is not built (exercise 4, p. 347). Page 14.3 sketches it; MLX's Metal sort is a merge sort of
          this shape to read.
        </li>
        <li>Batcher's odd-even merge sort network (p. 345) is named; only the bitonic network is stepped, on page 14.9.</li>
        <li>
          Sample sort (p. 346) gets one question. The book's advice to use it for data spread over several GPUs' memories is
          reported as the book's and not checked.
        </li>
        <li>MSD radix sort (p. 346) is described and used to explain why top-k is a selection; no MSD sort is built.</li>
        <li>
          Thrust (p. 347) is named and never run: neither the emulator nor Metal builds it, so the library table on 14.9
          gives the method names, unverified on hardware here.
        </li>
        <li>
          A one-kernel radix pass with a grid-wide barrier under a cooperative launch is not built. Neither is the
          single-pass OneSweep design the book cites (p. 336): Metal documents no forward progress between threadgroups.
        </li>
        <li>The chapter's references other than OneSweep (pp. 347–348) are not discussed.</li>
      </ul>
      <Beyond>
        <p>
          Which parts to keep short is our choice, from PLAN.md's tier for this chapter. The Metal fact is in
          notes/today/ch14-facts.md C4.
        </p>
      </Beyond>

      <h2>The project so far</h2>
      <p>
        Lab 14.3's sorted rows are the sorted candidates of milestone M5, sampling on the GPU: for every row of{' '}
        <code>[batch × vocab]</code> probabilities, the token ids largest first, ties by increasing id, and the
        probabilities in that order.{' '}
        <span data-lab143={loading ? 'loading' : own143.join(',') || 'none'}>
          {loading
            ? 'Reading reports.'
            : !ran
              ? 'No lab results are recorded on this machine yet.'
              : own143.length
                ? `Your own lab 14.3 passes on ${own143.map(where).join(' and ')} on this machine.`
                : 'Your own lab 14.3 does not pass on any backend of this machine yet.'}
        </span>{' '}
        See the <a href={href('ch14', 'labs')}>labs</a>.
      </p>
      <table className="data" data-table="m5">
        <thead>
          <tr><th>Piece of M5</th><th>Lab</th><th>Your own work passes on</th></tr>
        </thead>
        <tbody>
          {M5.map((m) => {
            const on = passingOn(reports, m.lab);
            return (
              <tr key={m.lab} data-m5={m.num} data-pass={loading ? 'loading' : on.join(',') || 'none'}>
                <td>{m.what}</td>
                <td className="nowrap">
                  <a href={href(m.ch, 'labs')}>lab {m.num}</a>
                </td>
                <td>{loading ? 'reading reports' : on.length ? on.map(where).join(', ') : <span className="muted">nothing yet</span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p>
        What is left of M5 is the sampler itself: module X1 joins these pieces, replaces the full sort with a selection for
        top-k, and draws the token. In the book, chapter 15 comes next and returns to matrix multiplication, the computation
        behind every linear layer, for milestone M3. Start at <a href={href('ch15')}>chapter 15</a>.
      </p>
    </>
  );
}
