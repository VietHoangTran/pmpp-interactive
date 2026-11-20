import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Beyond, Source } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { functionSource } from '../../sim/toolchain';

// The kernel for the first question is a snippet of our own, loaded at run time, so there
// is no copy to drift: labs/ch12/snippets/wrap_pairs.cu. Wrap.snippets.test.ts runs it on
// the course's emulator (where it passes, which is the point of the question) and pins the
// numbers below.
export const PAIRS_FILE = 'ch12/snippets/wrap_pairs.cu';

/**
 * Global traffic per input key of exercise 1's three kernels carrying (column, logit)
 * pairs: kernel 1 reads the logit and writes its flag (8), kernel 2 reads the flag and
 * writes the offset (8, not the scan's own traffic), kernel 3 reads logit, flag and offset
 * (12) and writes 8 bytes per kept pair. The column is the thread's index, never read.
 */
export const threeKernelPairBytes = (f: number) => 8 + 8 + 12 + 8 * f;
/** The same filter that reads each logit once and writes each kept pair once (lab 12.4's walk). */
export const onePassPairBytes = (f: number) => 4 + 8 * f;
export const PAIR_FRACTION = 0.25;

export const questionsWrap: Record<string, Question> = {
  'ch12/wrap/q-pairs': {
    kind: 'choice',
    id: 'ch12/wrap/q-pairs',
    prompt: (
      <>
        The kernel above collects the candidates of one row of logits as (column, logit) pairs, unstable. Both counters
        start at 0. Its test checks that the count is right and that every <code>outVal[k]</code> is the logit at column{' '}
        <code>outIdx[k]</code>. Where can that test fail?
      </>
    ),
    options: [
      {
        label: 'Nowhere: every kept column is written once, and so is every kept logit.',
        why: 'Each array on its own is a correct unstable filter. The pairs are the problem: slot k of one array and slot k of the other come from two different counters.',
      },
      {
        label:
          'Wherever another thread can take a slot between one thread\'s two atomics: on a GPU, or on the race emulator, which runs threads at the same time. Never on emu.',
        correct: true,
      },
      {
        label: 'On emu, under the reversed thread order.',
        why: 'Reversed changes which thread gets slot 0, but emu runs each thread to its end before the next, so every thread gets the same slot from both counters.',
      },
      {
        label: 'Only on AMD, where the compiler turns each atomic into one per wavefront.',
        why: 'Aggregation ranks the lanes of a wavefront in lane order at both counters, so inside one wavefront they agree. What can differ is the order in which warps reach each counter, on any vendor.',
      },
    ],
    explain: (
      <>
        <p>
          An atomic makes each update whole; it gives updates to one address no order, and updates to two addresses no
          order relative to each other. Thread A can take slot 3 of <code>idxCount</code> and slot 5 of{' '}
          <code>valCount</code>, so <code>outVal[3]</code> belongs to some other column. The counts stay right and every
          column and every logit is there: only the pairing breaks, which is why a test that compared the two arrays
          separately, as sorted lists, would pass.
        </p>
        <p>
          The course's emulator runs one thread until a barrier, a warp <code>_sync</code> call or its end, so nothing
          runs between a thread's two atomics: the unit test runs this kernel there in all three thread orders, and
          every pair matches. No tool names it either: every access is atomic or to a slot one thread owns, so there is
          no data race for ThreadSanitizer to report. The fix is one slot per pair: one <code>fetch_add</code>, then
          both stores at that slot.
        </p>
      </>
    ),
    source: 'ours; atomics on one address in no fixed order: notes/today/ch12-facts.md (settled table, ch09 A3); the emulator\'s schedule: labs/README.md; run by Wrap.snippets.test.ts',
  },
  'ch12/wrap/q-unstable-inplace': {
    kind: 'choice',
    id: 'ch12/wrap/q-unstable-inplace',
    prompt: (
      <>
        Fig. 12.2's kernel, the one atomic per kept key, is called with <code>output == input</code>. Can it compact
        the keys in place?
      </>
    ),
    options: [
      {
        label: 'Yes: a kept key never moves to a higher index, just as in the stable filter.',
        why: 'That holds for the stable filter, whose slot counts only the kept keys before it. A counter hands slots out in any order: thread 0 can be served last and get the highest slot.',
      },
      {
        label: 'Yes, if every block first reads its keys into shared memory and passes a barrier before taking slots.',
        why: 'That orders one block\'s reads before its own writes. Its slots can still land in a later block\'s tile, which may not have been read yet.',
      },
      {
        label: 'No: a thread can be handed a slot above its own index, where a key nobody has read yet may still live.',
        correct: true,
      },
      {
        label: 'Only where the compiler aggregates the atomic per warp.',
        why: 'Aggregation changes how many atomics reach the counter, not which slots come back: the warps still reach the counter in any order.',
      },
    ],
    explain: (
      <p>
        Section 12.7's rule, never write a slot before its old key has been read, is easy to keep in a stable filter
        because slot ≤ index: keys only move toward the front, so only earlier blocks' reads matter. An unstable filter
        promises no such bound. In place it would need every read in the grid done before any write, a grid-wide
        barrier, which exists only under a cooperative launch on CUDA and HIP and not at all on Metal. A stable filter
        is the cheaper way to compact in place.
      </p>
    ),
    source: 'ours, from ch. 12 §12.2, p. 291 (slots in arbitrary order) and §12.7, p. 300; grid barriers: ch12-facts settled table (ch04 A7, B11, C3)',
  },
  'ch12/wrap/q-chain': {
    kind: 'choice',
    id: 'ch12/wrap/q-chain',
    prompt: (
      <>
        A sampler filters a row of (token id, logit) pairs three times: it drops banned tokens with a stable filter,
        applies min-p with an unstable atomic filter because it is faster, then keeps the pairs at or above a top-k
        threshold with a stable filter. In what order are the final candidates?
      </>
    ),
    options: [
      {
        label: 'Column order: the last filter is stable.',
        why: 'A stable filter keeps the order of its input. Its input here is whatever order the atomic filter produced.',
      },
      {
        label: 'Column order, because the first filter was stable.',
        why: 'The unstable filter in the middle reorders what the first one kept.',
      },
      {
        label: 'Descending logit order: min-p and top-k both test the logit.',
        why: 'A filter only removes; it never sorts. Ordering by logit is chapter 14\'s job.',
      },
      { label: 'An order that can change from run to run.', correct: true },
    ],
    explain: (
      <p>
        Stability is a promise about a filter's own input, not about the original row: one unstable step anywhere makes
        the final order arbitrary. All three tests here look at one pair at a time, so the same set comes out of a
        single stable filter on the three tests ANDed together, in column order, with one read of the row instead of
        three. A test that depends on the order of the list, such as top-p's cumulative mass, could not be folded in
        that way.
      </p>
    ),
    source: 'ours, from ch. 12 §12.1, p. 290 (stable and unstable) and the notes\' "Relevance to LLM-inference kernels" (combining criteria)',
  },
  'ch12/wrap/q-rightalign': {
    kind: 'choice',
    id: 'ch12/wrap/q-rightalign',
    prompt: (
      <>
        You need a stable filter that packs the kept keys at the <em>end</em> of the array, in place, with the
        single-pass design of section 12.7. Which way must the chain between blocks run?
      </>
    ),
    options: [
      {
        label: 'Last block to first: each block waits for the block after it, which has read its keys and knows how many kept keys come after.',
        correct: true,
      },
      {
        label: 'First block to last, as in section 12.7.',
        why: 'That orders each block\'s writes after the earlier blocks\' reads. Here keys move toward the end, so a block writes over slots of later blocks.',
      },
      {
        label: 'No chain: keys that move outward never land on a key that has not been read.',
        why: 'They land on keys of later blocks, which may not have been read yet.',
      },
      {
        label: 'Either way, provided each block has a barrier between its reads and its writes.',
        why: 'The barrier orders reads and writes inside one block. Keys also cross into other blocks, and only the chain orders those.',
      },
    ],
    explain: (
      <p>
        A key's slot is now set by the kept keys <em>after</em> it, a scan from the end, and every key moves toward the
        end. A block's writes land in its own tile or in later ones, so no block may write before every later block has
        read. That is section 12.8's case of adding rows and columns: the one-directional chain runs from the block with
        the last keys to the block with the first, and on CUDA its dynamic numbers are handed out in that order.
      </p>
    ),
    source: 'ours, from ch. 12 §12.7, p. 300 and §12.8, p. 301 (the reversed constraint for keys that move outward)',
  },
  'ch12/wrap/q-pairbytes': {
    kind: 'numeric',
    id: 'ch12/wrap/q-pairbytes',
    prompt: (
      <>
        Exercise 1's three kernels again, now carrying pairs: each key is a 4-byte logit, each kept candidate is written
        as a (column, logit) pair of 8 bytes, and the column is the thread's index, never read. Flags and offsets are 4
        bytes. With a quarter of the logits kept, how many bytes of global traffic per input logit, not counting the
        scan's own traffic?
      </>
    ),
    answer: threeKernelPairBytes(PAIR_FRACTION),
    unit: 'bytes',
    explain: (
      <p>
        Kernel 1 reads the logit and writes the flag, 8; kernel 2 reads the flag and writes the offset, 8; kernel 3 reads
        the logit, the flag and the offset, 12, and writes 8 × 0.25 = 2 per input. 8 + 8 + 12 + 2 ={' '}
        {threeKernelPairBytes(PAIR_FRACTION)}. A kernel that reads each logit once and writes each kept pair once moves 4
        + 2 = {onePassPairBytes(PAIR_FRACTION)}, a fifth as much: this is why lab 12.4's block walks its row once and
        never writes a flag or an offset to memory.
      </p>
    ),
    source: 'ours, with the accounting of section 12.5 (exercise 1, p. 302); the pairs are lab 12.4\'s',
  },
};

/** The kernel of the first question, from the snippet file. */
function PairsKernel() {
  const file = useLabFile(PAIRS_FILE);
  const fn = file && !file.startsWith('// could not load') ? functionSource(file, 'candidatePairs') : null;
  return (
    <Code
      tabs={[
        {
          label: 'CUDA (not a lab file: for this question only)',
          code: fn ? fn.text : file ?? `// loading labs/${PAIRS_FILE}`,
          status: 'front-end-checked; wrong on purpose',
          tiers: { logits: 'global', outIdx: 'global', outVal: 'global', idxCount: 'global', valCount: 'global', v: 'reg', c: 'reg' },
        },
      ]}
      caption={
        <>
          From <code>labs/{PAIRS_FILE}</code>. The unit tests run it on the course's emulator.
        </>
      }
    />
  );
}

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const m5 = backends.filter((b) => labStatus(reports[b], 'ch12/lab4_candidates', 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch12/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul>
        <li>
          A filter answers one question per kept key: which output slot is mine? An unstable filter takes the next free
          slot from a counter, with an atomic. A stable one counts the kept keys before it, an exclusive scan of 0/1
          flags. Pay for stability when the input order means something: a sorted list, a tie rule, a seeded draw.
        </li>
        <li>
          On one counter, contention is the cost, so touch it less: once per warp with a vote (a leader, a popcount, a
          broadcast, and a binary prefix sum for each lane's rank), once per block with a private list in shared memory.
          Both also put neighbouring keys in neighbouring slots, so the stores coalesce.
        </li>
        <li>
          Warp code that ports: a ballot that every lane takes in uniform control flow, not{' '}
          <code>__activemask()</code> inside a branch; a mask with its own type, 64 bits on HIP, counted with the
          popcount of that type; the width read at run time. On Metal, <code>simd_sum</code> and{' '}
          <code>simd_prefix_exclusive_sum</code> are the count and the rank.
        </li>
        <li>
          A stable filter is keep, exclusive scan, scatter. Staging each block's survivors in shared memory writes
          contiguous runs; coarsening makes the runs longer and the scan cheaper. The bigger lever is passes over memory:
          at half kept, exercise 1's three kernels move 30 bytes per key, a count-then-scatter design 10, one pass 6.
        </li>
        <li>
          In place, keys only move toward the front, so no block may write before every earlier block has read. The
          single-pass scan's chain gives that only with a release, an acquire and a barrier after the reads, and only
          where waiting on an earlier block is documented: CUDA, with block numbers from a counter. On HIP and Metal,
          compact in place with one block (threadgroup) that walks its range in order, with barriers only; several
          dispatches are the documented-safe design out of place.
        </li>
        <li>
          When the index decides a key's position, no scan is needed, and the in-place chain runs the way the keys move.
          Libraries' filters are stable; frameworks that cannot return a shape that depends on the data mask instead of
          compacting, and a kernel that writes a list and a count is what lets the next kernel skip the losers.
        </li>
      </ul>
      <Source section="12.9" pages="302" />

      <h2>In code you will read</h2>
      <p>The same choices show in the GPU code of libraries you already call. Five places to look.</p>
      <Beyond>
        <p>
          Not in the book. Read in PyTorch's and llama.cpp's source and documentation on 2026-11-20
          (notes/today/ch12-facts.md D1, D2, D4); where a line says "by our reading", it is ours.
        </p>
      </Beyond>
      <ul data-list="real-code">
        <li>
          <b>Stable because a caller relies on it.</b> PyTorch's <code>torch.nonzero</code> promises that its result
          "is sorted lexicographically, with the last index changing the fastest (C-style)". On CUDA it is built on
          CUB's <code>DeviceSelect::Flagged</code>, a stable filter.
        </li>
        <li>
          <b>Order chosen over atomics.</b> <code>torch.topk</code>'s CUDA gather counts each warp's matches with a
          ballot, and its source says an atomicAdd-based reservation "would make the order depend on warp
          scheduling". That is section 12.2 against section 12.5, decided for determinism.
        </li>
        <li>
          <b>The ballot form of section 12.3.</b> llama.cpp's CUDA top-k places the keys tied at its threshold with a
          ballot and a masked popcount in uniform control flow, in column order; PyTorch's <code>topk</code> counts
          with a ballot too.
        </li>
        <li>
          <b>Staging for coalesced writes.</b> PyTorch's <code>nonzero_static</code> on CUDA is exercise 1 with
          coarsening, and before it scatters it rearranges its keys with a block exchange "for a slightly better
          coalescing for writes": section 12.6's idea in library code.
        </li>
        <li>
          <b>Deterministic is not stable.</b> llama.cpp's Metal top-k compacts each row in one threadgroup with
          SIMD-group sums and prefix sums, in a way that does not depend on scheduling. By our reading, though, each
          lane owns a strided substream and lanes are ranked before their elements, so the keys above the threshold
          come out in lane-major order, not column order: the same every run, and still not stable. What makes a
          compaction stable is ranking every kept key by the number of kept keys before it in input order, as Figs.
          12.8 to 12.10 all do; llama.cpp's main path ranks lanes before elements instead. (Its comment says it ranks
          "by column order"; its tie path does.)
        </li>
      </ul>

      <h2>A mixed check</h2>
      <p>
        These draw on the whole chapter and on situations the sections did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly on the first recorded try.` : ''}
      </p>
      <Beyond>
        <p>The kernel and the situations below are ours.</p>
      </Beyond>
      <PairsKernel />
      <Ask q={q['ch12/wrap/q-pairs']} label="Find the defect" />
      <Ask q={q['ch12/wrap/q-unstable-inplace']} />
      <Ask q={q['ch12/wrap/q-chain']} />
      <Ask q={q['ch12/wrap/q-rightalign']} />
      <Ask q={q['ch12/wrap/q-pairbytes']} label="Work it out" />

      <h2>The book's exercises</h2>
      <p>
        The chapter ends with four exercises (p. 302), all about the stable filter. Each is worked somewhere in this
        course; the book prints no answers, so the answers here are ours.
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
              <td>The stable filter as three kernels: keep flags, exclusive scan, output</td>
              <td>
                the keep-scan-scatter explorer and the listing in <a href={href('ch12', '12.5')}>12.5</a>, and part A of{' '}
                <a href={href('ch12', 'labs')}>lab 12.3</a>
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td>The stable filter as one kernel on a single-pass scan, and its advantages</td>
              <td>
                the byte count in <a href={href('ch12', '12.5')}>12.5</a>, the advantages question in{' '}
                <a href={href('ch12', '12.7')}>12.7</a>, and <a href={href('ch12', 'labs')}>lab 12.5</a> (optional, CUDA
                and HIP)
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td>Privatisation of exercise 2's kernel</td>
              <td>
                the staging explorer and scene in <a href={href('ch12', '12.6')}>12.6</a>; part B of{' '}
                <a href={href('ch12', 'labs')}>lab 12.3</a> in several kernels, lab 12.5 in one
              </td>
            </tr>
            <tr>
              <td>4</td>
              <td>Thread coarsening of exercise 3's kernel</td>
              <td>
                the coarsening questions in <a href={href('ch12', '12.6')}>12.6</a>; the coarse factor of{' '}
                <a href={href('ch12', 'labs')}>lab 12.3</a> part B and of lab 12.5
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Where the PDF looks wrong</h2>
      <p>
        The copy of the book this course was built from has a number of apparent slips in chapter 12, and a few
        listings that current documentation reads differently. They are listed so that you are not thrown by them when
        you read the book alongside. Where one changes code, the course's labs and snippets have a corrected, tested
        version.
      </p>
      <ul data-list="errata">
        <li>Fig. 12.1 captions both panels "(a)"; the second, the stable filter, should be (b).</li>
        <li>
          p. 292 cites "Chapter 5" for the way the hardware coalesces a warp's loads and stores. This edition teaches it
          in chapter 6, <a href={href('ch06', '6.1')}>section 6.1</a>.
        </li>
        <li>
          Figs. 12.3 and 12.4 declare <code>unsigned int j;</code> on line 08 and give it a value only in the leader, yet
          every lane passes it to the shuffle. Reading an uninitialised variable is undefined in C++, and the course's
          emulator build took that licence: every passing lane ran the leader's branch. The snippets and labs write{' '}
          <code>j = 0</code> (<a href={href('ch12', '12.3')}>12.3</a>).
        </li>
        <li>
          Fig. 12.3 treats <code>__activemask()</code> inside the branch as the set of lanes that passed. NVIDIA's guide
          now says it cannot be used that way: the kernel stays correct, but "one atomic per warp" is not guaranteed.
          The listing is also CUDA-only: HIP rejects its 32-bit shuffle mask, and widening only that leaves it wrong on
          64-lane wavefronts (<a href={href('ch12', '12.3')}>12.3</a>).
        </li>
        <li>
          Fig. 12.4 shows no header and no namespace for <code>coalesced_group</code>, and its <code>size()</code> is
          now a legacy alias of <code>num_threads()</code>.
        </li>
        <li>
          p. 294 says the compiler already coalesces atomics, and refrains when only one thread is active, without
          naming a compiler or version. NVIDIA says the first only in a blog post last updated in 2017; the second is in
          no NVIDIA source found.
        </li>
        <li>
          p. 295 prints <code>__ballot_sync</code> without parentheses and gives no parameters for any of the three
          votes it names.
        </li>
        <li>
          Fig. 12.6: the comment typo "intialize"; the shared <code>j</code> of line 26 reuses the name of the local{' '}
          <code>j</code> of line 19 and breaks the <code>_s</code> convention (lab 12.1 calls it <code>j_s</code>); and
          thread 0 makes its public atomic even when the block kept nothing. The output counter is said to be zeroed
          before the launch only for Fig. 12.2 (p. 291); Figs. 12.3, 12.4 and 12.6 need it too.
        </li>
        <li>
          Fig. 12.5 commits the blocks in the order green, orange, yellow, blue: one arbitrary order of many, which the
          text does not explain.
        </li>
        <li>
          Fig. 12.8 calls <code>gridExclusiveScan</code>, which is never defined, and reads <code>input[i]</code> with
          no <code>i &lt; N</code> check.
        </li>
        <li>
          p. 300 says thread 1 must finish "reading k0" before thread 3 writes k3; by the sentence's own logic it is k1.
          The example cites Fig. 12.10, where four threads handle eight keys, so its thread numbers do not map onto keys
          (<a href={href('ch12', '12.7')}>12.7</a>).
        </li>
        <li>
          p. 301: "earlier thread blocks write to memory locations previously occupied by keys read by earlier thread
          blocks"; the second "earlier" should be "later", as the next sentence confirms. The page also has "in the
          section" and "more complex data in-place data movement patterns".
        </li>
        <li>
          Reference [4], the CUB link on p. 302, returns 404; the pages now live under{' '}
          <code>nvidia.github.io/cccl/unstable/</code>.
        </li>
      </ul>
      <Aside>
        <p>
          These are our readings of one PDF, checked against current vendor documentation on 2026-11-20
          (notes/today/ch12-facts.md A1, A5, A7, A8, B1). If your copy differs, trust your copy.
        </p>
      </Aside>
      <p>
        Some of the book is left out on purpose, so that you can tell an omission from a gap. The heap compaction of
        §12.1 (pp. 289–290) motivates the chapter, but no lab builds an allocator. In-place transposition of matrices and
        tensors (§12.8, p. 301, reference [2]) is named, not covered. Fig. 12.4's cooperative-groups kernel is shown and
        run on the emulator, not built in a lab: Metal has no cooperative groups, and lab 12.1's vote covers the same
        steps on every backend. Fig. 12.8's undefined scan is not reproduced; labs 12.3 and 12.5 implement it two ways.
        Exercises 2 to 4 as one kernel are done in full only in the optional lab 12.5. Thrust and CUB, and their
        counterparts on AMD and Apple, are described with their guarantees, not called. References [1] and [2] are cited
        but not checked.
      </p>

      <h2>The project so far</h2>
      <p>
        Lab 12.4's candidate compaction is a piece of milestone M5 of the through-line project, sampling on the GPU:
        stable (column, logit) pairs and a count for every row of <code>[batch × vocab]</code> logits, the step between a
        threshold and a draw. Lab 9.3 gave M5 its token counts; module X1 adds lab 10.3's row max for the min-p
        threshold, chapter 14's sort for top-k and top-p, and the draw itself.{' '}
        <span data-m5={m5.join(',') || 'none'}>
          {backends.length === 0
            ? 'No lab results are recorded on this machine yet.'
            : m5.length
              ? `Your own lab 12.4 passes on ${m5.map(where).join(' and ')} on this machine.`
              : 'Your own lab 12.4 does not pass on any backend of this machine yet.'}
        </span>{' '}
        See the <a href={href('ch12', 'labs')}>labs</a>.
      </p>
      <p>
        Chapter 13 runs the other way. The book calls filtering the inverse of merging (p. 289): a merge interleaves two
        sorted lists into one, in order. It is the first pattern in which a thread cannot work out its inputs from its
        index, and a building block of chapter 14's sorting, which gives M5 its sorted candidates. Start at{' '}
        <a href={href('ch13')}>chapter 13</a>.
      </p>
    </>
  );
}
