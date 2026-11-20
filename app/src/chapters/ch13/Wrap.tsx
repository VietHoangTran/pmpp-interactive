import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';

// Chapter 13's summary page: what to carry forward, a mixed check on situations the
// sections did not show, where the book's four exercises are worked, the PDF's apparent
// slips, what the condensed chapter left out, and the project status. No kernel is shown
// for a question here, so there is no snippet to tie.

// Exercise 4 (p. 328), computed rather than typed.
const EX4 = { m: 1_030_400, n: 608_000, per: 8, block: 1024 };
const EX4_THREADS = (EX4.m + EX4.n) / EX4.per; // 204,800
const EX4_BLOCKS = EX4_THREADS / EX4.block; // 200
// The input-split question.
const SPLIT = { m: 1000, n: 1000, threads: 4 };
const SPLIT_BUSIEST = SPLIT.m / SPLIT.threads + SPLIT.n; // 1,250
const SPLIT_EVEN = (SPLIT.m + SPLIT.n) / SPLIT.threads; // 500
// The top-k question: the lab's basic kernel with total < m + n.
const TOPK = { len: 64, k: 50, per: 8, block: 128 };
const TOPK_WORKING = Math.ceil(TOPK.k / TOPK.per); // 7

const n = (v: number) => v.toLocaleString('en-US');

export const questionsWrap: Record<string, Question> = {
  'ch13/wrap/q-input-split': {
    kind: 'numeric',
    id: 'ch13/wrap/q-input-split',
    prompt: (
      <>
        A colleague splits a merge by input instead of by output. A and B each have {n(SPLIT.m)} elements, and every
        element of B is smaller than A[0]. Each of {SPLIT.threads} threads takes {SPLIT.m / SPLIT.threads} consecutive
        elements of A, binary-searches B for the elements that belong between them, and merges. How many outputs does the
        busiest thread write?
      </>
    ),
    answer: SPLIT_BUSIEST,
    unit: 'outputs',
    explain: (
      <p>
        All of B comes before A[0], so the thread that owns A[0..{SPLIT.m / SPLIT.threads}) also gets every element of B:{' '}
        {SPLIT.m / SPLIT.threads} + {n(SPLIT.n)} = {n(SPLIT_BUSIEST)} outputs, while the other three write{' '}
        {SPLIT.m / SPLIT.threads} each. The whole merge waits for that one thread. Splitting the output gives every
        thread {SPLIT_EVEN}, whatever the data; only the input slices vary. That is the reason for the chapter's
        output-first decomposition.
      </p>
    ),
    source: 'ch. 13 §13.3, pp. 306–308; ch. 22 §22.2, p. 535 (load balance); the input-split variant is ours',
  },
  'ch13/wrap/q-descending': {
    kind: 'choice',
    id: 'ch13/wrap/q-descending',
    prompt: (
      <>
        Two top-k lists arrive sorted largest first, and you want their merge largest first, stable, with ties going to
        A. What has to change in <code>co_rank</code> and <code>merge_sequential</code>?
      </>
    ),
    options: [
      {
        label: (
          <>
            Only the merge's comparison, which becomes <code>A[i] &gt;= B[j]</code>; the search stays.
          </>
        ),
        why: 'The search would still look for the split of an ascending merge: its tests would call i too high exactly when it is too low.',
      },
      {
        label: 'Every comparison flips direction and keeps its strictness, in the merge and in both tests.',
        correct: true,
      },
      {
        label: 'Reverse both lists, merge them ascending, and reverse the merged result.',
        why: "The keys come out in the right order and each list keeps its own order, since it is reversed twice. But the ascending merge's A-first tie rule ends up reversed, so on equal keys B's element now comes first. That is not the stable merge you asked for, unless the ascending merge sends ties to B.",
      },
      {
        label: 'Swap the roles of A and B in both the search and the merge.',
        why: 'That only sends ties to B. The keys would still come out smallest first.',
      },
    ],
    explain: (
      <p>
        In a descending merge "comes first" means "is larger", so every comparison flips:{' '}
        <code>A[i] &gt;= B[j]</code> in the merge, <code>A[i-1] &lt; B[j]</code> and <code>B[j-1] &lt;= A[i]</code> in
        the search. The tie rule must stay the same in all three places: the merge takes A on equal keys, the first test
        does not call i too high on a tie, the second does call j too high on one. Negating the keys and merging
        ascending gives the same result. Merging per-block top-k lists this way is how llama.cpp's Metal top-k works for
        small k.
      </p>
    ),
    source: 'ch. 13 §13.4, p. 311 (the tie rule in the tests); notes/today/ch13-facts.md D3; the descending case is ours',
  },
  'ch13/wrap/q-topk-threads': {
    kind: 'numeric',
    id: 'ch13/wrap/q-topk-threads',
    prompt: (
      <>
        Lab 13.1's basic kernel merges two sorted lists of {TOPK.len} elements each, but you want only the first{' '}
        <code>total = {TOPK.k}</code> outputs. With {TOPK.per} outputs per thread and blocks of {TOPK.block} threads, how
        many threads run a co-rank search?
      </>
    ),
    answer: TOPK_WORKING,
    unit: 'threads',
    explain: (
      <p>
        The launch is sized for <code>total</code>, not m + n: ⌈{TOPK.k} / ({TOPK.per} × {TOPK.block})⌉ = 1 block. Thread t
        owns outputs from {TOPK.per}t, so threads 0 to {TOPK_WORKING - 1} start below {TOPK.k} (thread {TOPK_WORKING - 1}{' '}
        owns only outputs {TOPK.per * (TOPK_WORKING - 1)} and {TOPK.per * (TOPK_WORKING - 1) + 1}, because its{' '}
        <code>k_next</code> is clamped to {TOPK.k}). Threads {TOPK_WORKING} to {TOPK.block - 1} start at or past{' '}
        <code>total</code> and return before searching. {TOPK_WORKING} threads, two searches each, and nothing past{' '}
        <code>C[{TOPK.k - 1}]</code> is ever written.
      </p>
    ),
    source: 'lab 13.1 (the total mode is ours); ch. 13 Fig. 13.9, p. 313',
  },
  'ch13/wrap/q-ex4': {
    kind: 'choice',
    id: 'ch13/wrap/q-ex4',
    prompt: (
      <>
        The book's exercise 4: arrays of {n(EX4.m)} and {n(EX4.n)} elements, {EX4.per} outputs per thread, {n(EX4.block)}{' '}
        threads per block. How many threads binary-search global memory in the basic kernel (Fig. 13.9); and in the
        tiled kernel (Figs. 13.11 to 13.13), how many search global memory and how many search shared memory?
      </>
    ),
    options: [
      { label: `${n(EX4_THREADS)}; then ${n(EX4_BLOCKS)} and ${n(EX4_THREADS)}.`, correct: true },
      {
        label: `${n(EX4_THREADS)}; then ${n(EX4_THREADS)} and 0.`,
        why: 'In Fig. 13.11 only thread 0 of each block runs the block-level co_rank on global memory, and every thread searches the tiles.',
      },
      {
        label: `${n(EX4.m + EX4.n)}; then ${n((EX4.m + EX4.n) / EX4.block)} and ${n(EX4.m + EX4.n)}.`,
        why: 'That is one thread per output. Each thread here produces 8.',
      },
      {
        label: `${n(EX4_THREADS)}; then ${n(EX4_BLOCKS)} and ${n(EX4_BLOCKS)}.`,
        why: 'Lines 41 and 43 run in every thread, on the shared tiles, and so does line 48.',
      },
    ],
    explain: (
      <p>
        {n(EX4.m)} + {n(EX4.n)} = {n(EX4.m + EX4.n)} outputs at {EX4.per} per thread is {n(EX4_THREADS)} threads. In the
        basic kernel each of them calls <code>co_rank</code> twice on global memory. In the tiled kernel those threads
        form {n(EX4_BLOCKS)} blocks of {n(EX4.block)}; only thread 0 of each searches global memory, and every thread
        searches the shared tiles, three times per iteration counting line 48. The count assumes, as the exercise
        suggests, that 8 outputs per thread fixes the number of threads; it gives no <code>tile_size</code>. The
        libraries go further: a separate partition kernel makes one global search per tile boundary.
      </p>
    ),
    source: 'ch. 13 exercise 4, p. 328; Figs. 13.9, 13.11, 13.13',
  },
  'ch13/wrap/q-cache': {
    kind: 'choice',
    id: 'ch13/wrap/q-cache',
    prompt: (
      <>
        On some device the basic kernel runs about as fast at 8 outputs per thread as at 1, although by page 13.3's count
        each of its stores touches 8 times as many 32-byte segments at 8. What is the likeliest reason?
      </>
    ),
    options: [
      {
        label: 'Over eight steps each lane fills its own segment, and a cache can combine those stores.',
        correct: true,
      },
      {
        label: 'The count is wrong: at 8 outputs per thread neighbouring lanes still store to neighbouring addresses.',
        why: 'Lane t stores to C[8t + s] at step s: 32 bytes apart for 4-byte keys, one segment each. The count is right for one instruction.',
      },
      {
        label: 'At 8 outputs per thread the kernel became compute-bound, so memory no longer matters.',
        why: 'A merge does one comparison per output and no arithmetic on the data; it stays limited by memory and by its branches.',
      },
      {
        label: 'The device coalesces accesses from different warps into one request.',
        why: 'Coalescing combines the accesses of one warp\'s instruction. Merging later traffic is what a cache does, and it is not guaranteed.',
      },
    ],
    explain: (
      <p>
        The per-instruction count asks which segments one store of one warp touches, and it assumes nothing remembers
        anything between instructions. Each thread walks its own slice in order, so over eight steps each lane fills
        its own segment completely, and the bytes that must reach memory are the same as at one output per thread; only
        the number of separate requests differs, and a cache that keeps the line can absorb much of that. The searches
        that coarsening saves pay for some of the rest. Lab 13.1's first prediction measures it on your device. Predict
        from the model, then measure.
      </p>
    ),
    source: 'ch. 13 §13.5, p. 314, and §13.8, p. 327; the cache reasoning is ours (page 13.3, "What it costs")',
  },
};

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = realBackends(reports);
  const lab = 'ch13/lab1_merge';
  const passing = [...backends, 'emu'].filter((b) => labStatus(reports[b], lab, 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch13/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul>
        <li>
          When which inputs a thread needs depends on the data, split the output, not the input. Every thread gets the
          same number of outputs, and a search finds the input ranges that produce them.
        </li>
        <li>
          The co-rank is that search: a binary search over the splits i + j = k, steered by the elements on either side
          of the cut, with the bounds tests ahead of the reads. It takes about log2 N steps. The libraries call it merge
          path or merge partition.
        </li>
        <li>
          The search and the sequential merge must break ties the same way, or the merge stops being stable. Only a test
          with tagged values, such as key–value pairs checked against <code>std::merge</code>, can see the difference.
        </li>
        <li>
          Every thread pays for its searches, so give each thread many outputs. In the basic kernel that spreads each
          warp's accesses apart; the per-instruction model and the measurement can disagree, because caches see more
          than one instruction.
        </li>
        <li>
          Stage data-dependent reads through shared memory: coalesced loads in, irregular accesses there. When a tile
          cannot know in advance which inputs it will use, either load for the worst case and waste half (the book), or
          search first and load exactly (every tiled merge read for this course: CUB, rocPRIM, MLX, PyTorch MPS;
          llama.cpp's Metal merge does not stage at all).
        </li>
        <li>
          Never take <code>ceil</code> of an integer quotient. Write <code>(a + b - 1) / b</code>.
        </li>
      </ul>
      <Source section="13.9" pages="327" />

      <h2>A mixed check</h2>
      <p>
        These draw on the whole chapter and on situations the pages did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
      </p>
      <Beyond>
        <p>
          The input split, the descending merge, the top-k launch and the cache question are ours. Exercise 4 is the
          book's.
        </p>
      </Beyond>
      <Ask q={q['ch13/wrap/q-input-split']} label="Work it out" />
      <Ask q={q['ch13/wrap/q-descending']} />
      <Ask q={q['ch13/wrap/q-topk-threads']} label="Work it out" />
      <Ask q={q['ch13/wrap/q-ex4']} label="Book exercise 4" />
      <Ask q={q['ch13/wrap/q-cache']} />

      <h2>The book's exercises</h2>
      <p>
        The chapter ends with four exercises (pp. 327–328). The book prints no answers, so the answers in this course are
        ours.
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
              <td>The co-ranks of C[8] in the running example</td>
              <td>
                the co-rank explorer in <a href={href('ch13', '13.3')}>13.3</a>: set k = 8 and run the search
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td>Finish the co-rank search for thread 2 of the three-thread example</td>
              <td>
                the explorer in <a href={href('ch13', '13.3')}>13.3</a> at k = 6 (it stops at once), and part 5 of the
                animation "Searching for the split", which shows every thread's split
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td>Load only the elements an iteration will use, with an extra co-rank call</td>
              <td>
                "Load exactly what a tile uses" in <a href={href('ch13', '13.3')}>13.3</a>; part (b) of{' '}
                <a href={href('ch13', 'labs')}>lab 13.1</a>, in the shape the libraries ship; the sketch below for the
                book's own loop
              </td>
            </tr>
            <tr>
              <td>4a, 4b, 4c</td>
              <td>Threads that binary-search global and shared memory in the two kernels</td>
              <td>the "Book exercise 4" question above</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        Exercise 3 in the book's own loop: before the load loops, one thread computes the co-rank of the end of this
        iteration's outputs on global memory, <code>co_rank(min(C_curr + C_completed + tile_size, C_next), A, m, B, n)</code>,
        and shares it through shared memory behind a barrier. Subtracting <code>A_curr + A_consumed</code> gives the
        number of A elements this iteration needs, and the rest of the tile comes from B. Those two counts bound the load
        loops and become the lengths in the per-thread co-rank calls, and <code>A_consumed</code> is set from the result
        directly, so line 48 goes. Each iteration pays one more search on global memory and one more barrier, and loads
        half as much. Kept as two arrays of <code>tile_size</code> each, the tiles still need room for the worst case;
        packed into one array, A's part and then B's as lab 13.1 does, they need half the space.
      </p>
      <Source section="13.6, 13.10" pages="316, 328" />

      <h2>Where the PDF looks wrong</h2>
      <p>
        The copy of the book this course was built from has a number of apparent slips in chapter 13, most of them in the
        circular-buffer listings that this course leaves out. They are listed so that you are not thrown by them when
        you read the book alongside. The first eight are slips of wording or numbering; most of the rest affect code.
      </p>
      <ul data-list="errata">
        <li>p. 306 says the second part of the function "is on the right Fig. 13.2"; the listing is a single column.</li>
        <li>
          Fig. 13.3(b) labels the C[6] case i = 4, j = 2. The text and p. 308 give co-ranks 5 and 1, which is what the
          tie rule and Fig. 13.5 give (<a href={href('ch13', '13.3')}>13.3</a>).
        </li>
        <li>
          p. 309 prints thread t's slices as "A[i_t] to A[i_t + 1]", the + 1 outside the subscript. It means from A[i_t]
          up to just before A[i_(t+1)], and the same for B.
        </li>
        <li>
          In Fig. 13.5 (and Fig. 13.19) the comment on the <code>j_low</code> line reads <code>i_low = max(0,k-m)</code>,
          and the comment on the delta line has unbalanced parentheses. The course's copy fixes both comments.
        </li>
        <li>p. 311 says "In Fig. 13.7, for iteration 0"; iteration 0 is Fig. 13.6.</li>
        <li>Fig. 13.8's caption says "thread 0"; the text and Figs. 13.6 and 13.7 are about thread 1.</li>
        <li>
          p. 314 lists the basic kernel's first reads as "A[0], A[2], and B[0]". By the listing, thread 0 compares A[0]
          with B[0], thread 1 (no B elements) reads A[2], and thread 2 (no A elements) reads B[1].
        </li>
        <li>
          p. 315 says thread 1's A slice in the Fig. 13.8 example is much smaller than its B slice; there thread 1 takes
          three elements of A and none of B. p. 321 cites the tiled kernel as "Figs. 13.12 through 13.14" (it is Figs.
          13.11 to 13.13), and p. 322 calls it <code>merge_tile_kernel</code> once.
        </li>
        <li>
          <code>ceil</code> of an integer quotient in Fig. 13.9 line 03, Fig. 13.11 lines 05 and 06, and Fig. 13.12 line
          21: the division truncates first, so <code>ceil</code> changes nothing. With the book's own numbers the tiled
          kernel runs 3 iterations where p. 317 says 4, and outputs go unwritten (
          <a href={href('ch13', '13.3')}>13.3</a>).
        </li>
        <li>
          Fig. 13.13 line 48 counts <code>A_consumed</code> as if both tiles were full. The book allows that it is wrong
          after the last iteration (p. 320). Our own emulation shows it also goes wrong on an earlier iteration with a
          partly loaded tile; the circular kernel's line 50, with <code>min</code> lengths, is correct (
          <a href={href('ch13', '13.3')}>13.3</a>).
        </li>
        <li>
          The second <code>__syncthreads()</code> of Fig. 13.11 (line 16) is not explained. Our reading: every thread must
          have read <code>A_S[0]</code> and <code>A_S[1]</code> before the tile loads overwrite them.
        </li>
        <li>
          p. 322 says <code>A_S_consumed</code> and <code>B_S_consumed</code> start at 0; Fig. 13.16 starts them at{' '}
          <code>tile_size</code>, which the first fill needs.
        </li>
        <li>
          Fig. 13.16 line 37 computes B's destination slot with <code>A_S_consumed</code>; <code>B_S_consumed</code> is
          meant.
        </li>
        <li>
          The circular kernel has no <code>__syncthreads()</code> between its loads (Fig. 13.16, ends at line 39) and the
          searches (Fig. 13.18, from line 40), where the tiled kernel has one. One is needed for the same reason.
        </li>
        <li>
          Fig. 13.16's refill reads from <code>A[A_curr + A_consumed + i + threadIdx.x]</code>, which reloads elements
          still in the buffer. In our own emulation, adding <code>tile_size − A_S_consumed</code> to the source offset and
          the guard made the merges correct; Fig. 13.15(c) draws the refill starting after the previous tile, which
          agrees.
        </li>
        <li>
          Fig. 13.19 computes <code>j_m_1_cir</code> from <code>B_S_start+i-1</code>; <code>j-1</code> is meant. Several
          of its lines and one in Fig. 13.20 have an unmatched <code>)</code>, and neither listing has line numbers.
        </li>
        <li>
          Fig. 13.18 line 48 calls <code>merge_sequetial_circular</code>. Fig. 13.17(b) labels the B arrow "B_Start",
          and pp. 323–324 write <code>A_curr</code> and <code>B_curr</code> where the thread-level{' '}
          <code>a_curr</code> and <code>b_curr</code> are meant.
        </li>
      </ul>
      <p>And what the listings leave to you, which a real build needs:</p>
      <ul data-list="gaps">
        <li>
          <code>co_rank</code> and <code>merge_sequential</code> carry no qualifier. On CUDA and HIP they need{' '}
          <code>__device__</code>; an unqualified function is host-only.
        </li>
        <li>
          Fig. 13.9 clamps <code>k_next</code> but not <code>k_curr</code>, and relies on the launch to keep threads inside
          C. Clamp both, and launch nothing for an empty merge: HIP rejects a grid of zero blocks.
        </li>
        <li>
          <code>extern __shared__ int shareAB[]</code> has no size in the kernel; it comes from the third launch argument,
          which the chapter does not show.
        </li>
        <li>
          Integer <code>min</code> differs between backends: CUDA has mixed signed and unsigned overloads that return{' '}
          <code>unsigned</code>, HIP's header has none, and MSL wants both arguments of one type. The listings' ternaries
          avoid the question.
        </li>
      </ul>
      <Aside>
        <p>
          These are our readings of one PDF (notes/ch13.md, "Unclear in the PDF"), with the vendor points checked on
          2026-11-20 (notes/today/ch13-facts.md A4, A5, B1, C1; ch02-facts). Items marked as our emulation are inferences,
          not the book's. If your copy differs, trust your copy.
        </p>
      </Aside>

      <h2>What this chapter left out</h2>
      <p>
        The chapter is condensed in this course, so here is what is not covered and where to find it. §13.7, the
        circular-buffer kernel (pp. 321–327, Figs. 13.15 to 13.20), gets one paragraph on page 13.3 and its slips above:
        no interactive, no code and no lab, because no library or project read for this course uses one. The
        line-by-line walk through the tiled kernel (Figs. 13.11 to 13.13, pp. 316–320) is summarised on page 13.3; its
        bookkeeping, <code>A_consumed</code>, <code>C_completed</code> and the tail guards, lives in the loaded-versus-used
        explorer instead, and the lab teaches the exact-load version. Observation 1's two cases (pp. 306–307) are folded
        into Observation 2, and the sequential merge's trace (p. 306) into a paragraph on the start page. Map-reduce and
        Hadoop (p. 305) and set intersection and union (p. 303) are mentioned by the book in passing and not here.
      </p>
      <p>Nothing later in the course depends on what was left out.</p>

      <h2>The project so far</h2>
      <p>
        This chapter adds no milestone to the through-line project: nothing on the main path of an inference step, its
        matrix products, attention or quantised arithmetic, merges two sorted lists. The merge does sit under sampling,
        though. Lab 13.1's <code>total</code> mode is the step that combines per-block top-k lists, and the sampling
        module of this course can reuse it. Lab 13.1 is optional.{' '}
        <span data-lab131={passing.join(',') || 'none'}>
          {backends.length === 0 && !reports['emu']
            ? 'No lab results are recorded on this machine yet.'
            : passing.length
              ? `Your own version passes on ${passing.map(where).join(' and ')} on this machine.`
              : 'Your own version has not passed on this machine yet.'}
        </span>{' '}
        See the <a href={href('ch13', 'labs')}>lab</a>.
      </p>
      <p>
        Chapter 14 turns the merge into a sort. Its merge sort is this chapter's merge applied stage after stage, and the
        co-rank decides where each block's share of a merge starts. Most of that chapter, though, is radix sort, which
        compares no keys at all: each pass places every key with an exclusive scan from chapter 11. Start at{' '}
        <a href={href('ch14', 'start')}>chapter 14</a>.
      </p>
    </>
  );
}
