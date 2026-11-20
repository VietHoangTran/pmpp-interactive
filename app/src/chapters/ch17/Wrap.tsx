import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Beyond, Source, Today, type Tier } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { warpAccess } from '../../sim/coalesce';
import { EXERCISE1, FIG17_14, spmvRef, toJDS } from '../../sim/sparse';

// Chapter 17's summary page: what to carry forward, exercise 5's JDS kernel (ours, loaded
// from labs/ch17/snippets/spmv_jds.cu), a mixed check on situations the pages did not show,
// where the book's five exercises are worked, the PDF's apparent slips, what the condensed
// chapter left out, and the project status. Wrap.snippets.test.ts runs the kernel on the
// emulator and pins every number below.

// ---------------------------------------------------------------- exercise 5's kernel
export const JDS_SNIPPET = 'ch17/snippets/spmv_jds.cu';
export const JDS_LOOP_LINE =
  'for (unsigned int t = 0; t < jdsMatrix.numIters && r < jdsMatrix.iterPtr[t + 1] - jdsMatrix.iterPtr[t]; ++t) {';
export const JDS_INDEX_LINE = 'unsigned int i = jdsMatrix.iterPtr[t] + r;';
export const JDS_STORE_LINE = 'y[jdsMatrix.row[r]] += sum;';
const JDS_STATUS = 'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware';
// `value` is left uncoloured: it names both a local float and the struct's device pointer.
const JDS_TIERS: Record<string, Tier> = { x: 'global', y: 'global', sum: 'reg', r: 'reg', t: 'reg', i: 'reg', col: 'reg' };

/** The file without its leading comment block, which the caption replaces. */
export function snippetBody(src: string): string {
  const lines = src.split('\n');
  let k = 0;
  while (k < lines.length && (lines[k].startsWith('//') || lines[k].trim() === '')) k++;
  return lines.slice(k).join('\n');
}
const linesWith = (text: string, needles: readonly string[]): number[] =>
  text.split('\n').flatMap((l, i) => (needles.some((n) => l.trim() === n) ? [i + 1] : []));

function JdsKernel() {
  const src = useLabFile(JDS_SNIPPET);
  const code = src ? snippetBody(src) : '// loading labs/' + JDS_SNIPPET;
  return (
    <Code
      tabs={[{
        label: 'spmv_jds.cu (ours)',
        code,
        tiers: JDS_TIERS,
        status: JDS_STATUS,
        highlight: src ? linesWith(code, [JDS_LOOP_LINE, JDS_INDEX_LINE, JDS_STORE_LINE]) : [],
      }]}
      caption={
        <>
          Exercise 5, answered in <code>labs/ch17/snippets/spmv_jds.cu</code>. The book shows no JDS kernel; this one and
          its <code>JDSMatrix</code> struct are ours. Highlighted: the loop test that stops thread <code>r</code> once
          fewer than <code>r + 1</code> rows reach iteration <code>t</code>, the index of its element, and the store into
          the original row. Run on the course's emulator in three thread orders, on Fig. 17.14, exercise 1's matrix, a
          skewed matrix and one with empty rows.
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- numbers the questions use
const J14 = toJDS(FIG17_14);
const ONES6 = Array(6).fill(1);
/** Fig. 17.14's row sums with x all ones: what y should hold, starting from zero. */
export const ROW_SUMS = spmvRef(FIG17_14, ONES6, Array(6).fill(0));
/** y when the result goes to the sorted position r instead of row[r]. */
export const UNPERMUTED = J14.row.map((orig) => ROW_SUMS[orig]);
export const UNPERMUTED_RIGHT = UNPERMUTED.filter((v, r) => v === ROW_SUMS[r]).length;
/** Threads at work in each iteration of Fig. 17.14's JDS: [6, 5, 3, 1]. */
export const PER_ITER = J14.iterPtr.slice(1).map((p, t) => p - J14.iterPtr[t]);
const J_EX1 = toJDS(EXERCISE1);

/** One warp over uniform rows of 16 floats: CSR's sector requests and distinct sectors, and ELL's. */
const LEN = 16;
const WARP = 32;
const csrIter = (t: number) => warpAccess(Array.from({ length: WARP }, (_, l) => (LEN * l + t) * 4), 4, 32);
const ellIter = (t: number) => warpAccess(Array.from({ length: WARP }, (_, l) => (t * WARP + l) * 4), 4, 32);
const ITERS = Array.from({ length: LEN }, (_, t) => t);
export const SECTORS = {
  csrRequests: ITERS.reduce((n, t) => n + csrIter(t).transactions, 0),
  csrDistinct: new Set(ITERS.flatMap((t) => csrIter(t).segments)).size,
  ellPerIter: ellIter(0).transactions,
  ellDistinct: new Set(ITERS.flatMap((t) => ellIter(t).segments)).size,
};

/** A decode batch padded to its longest sequence. */
export const RAGGED = { seqs: 64, short: 100, long: 2000 };
const RAGGED_REAL = (RAGGED.seqs - 1) * RAGGED.short + RAGGED.long;
const RAGGED_SLOTS = RAGGED.seqs * RAGGED.long;
export const RAGGED_PCT = (100 * RAGGED_REAL) / RAGGED_SLOTS;

const n = (v: number) => v.toLocaleString('en-US');
const list = (xs: readonly number[]) => '[' + xs.join(', ') + ']';

export const questionsWrap: Record<string, Question> = {
  'ch17/wrap/q-jds-unpermuted': {
    kind: 'numeric',
    id: 'ch17/wrap/q-jds-unpermuted',
    prompt: (
      <>
        The kernel above on Fig. 17.14, with <code>x</code> all ones and <code>y</code> starting at zero, except that its
        last line writes <code>y[r] += sum</code>, the sorted position, instead of <code>y[jdsMatrix.row[r]]</code>. How
        many of the six entries of <code>y</code> come out right?
      </>
    ),
    answer: UNPERMUTED_RIGHT,
    unit: 'entries',
    explain: (
      <p>
        Thread <code>r</code> sums sorted row <code>r</code>, which is original row <code>row[r]</code>, and{' '}
        <code>row = {list(J14.row)}</code>. Writing to <code>y[r]</code> lands each sum in the wrong place unless{' '}
        <code>row[r] = r</code>, and only <code>r = 4</code> qualifies: original row 4 also sorts fifth. With a to o
        standing for 1 to 15, the six row sums are {list(ROW_SUMS)}, all different, so no other entry is right by
        coincidence: the defect gives {list(UNPERMUTED)}. The emulator run of this page's checked copy agrees, 5 of 6
        wrong. Sorting rows reorders the equations, so it is <code>y</code> that has to be put back.
      </p>
    ),
    source: 'ch. 17 §17.6, pp. 416–417, Fig. 17.14 (values ours); exercise 5, p. 422',
  },
  'ch17/wrap/q-transpose': {
    kind: 'choice',
    id: 'ch17/wrap/q-transpose',
    prompt: (
      <>
        A is stored in CSR, and a backward pass needs <code>Aᵀ·g</code>, where <code>g</code> has one entry per row of A.
        Which plan follows from the chapter?
      </>
    ),
    options: [
      {
        label: (
          <>
            Read A's CSR arrays as the CSC of Aᵀ: one thread per row of A loads <code>g[row]</code> once and adds{' '}
            <code>value·g[row]</code> into <code>out[col]</code> with an atomic, as SpMV/CSC does. If the product repeats,
            build the CSR of Aᵀ once and run the per-row kernel with no atomics.
          </>
        ),
        correct: true,
      },
      {
        label: (
          <>
            Run the CSR kernel unchanged with <code>g</code> in place of <code>x</code>.
          </>
        ),
        why: 'That computes A·g, which would need one entry of g per column of A. It is the wrong product.',
      },
      {
        label: (
          <>
            Swap <code>rowPtrs</code> and <code>colIdx</code>, then run the CSR kernel.
          </>
        ),
        why: 'rowPtrs holds m + 1 offsets, colIdx holds z column numbers: different lengths and different meanings. Grouping the entries by column takes a pass over them.',
      },
      {
        label: 'Store A as ELL: its column-major layout already is the transpose.',
        why: 'ELL turns the storage order of the padded rectangle, not the matrix. Slot t·stride + r still belongs to row r.',
      },
    ],
    explain: (
      <p>
        CSC is CSR with rows and columns swapped (p. 418), so A's CSR arrays are exactly the CSC of Aᵀ. The kernel that reads
        them one column of Aᵀ per thread is SpMV/CSC, with its one virtue, <code>g</code> loaded once and coalesced, and
        its costs: atomics on the output and divergence from row lengths. When the product repeats, the book's advice to
        keep a second, column-oriented copy (p. 409) pays, and building it is exercise 3's count, scan and scatter, keyed
        by column. PyTorch's FlexAttention does the same for its backward pass: it generates <code>q_indices</code>, a
        transposed copy of its block mask.
      </p>
    ),
    source: 'ch. 17 §17.3, p. 409; §17.7, pp. 418–421; ch17-facts A9, D3; the backward-pass setting is ours',
  },
  'ch17/wrap/q-csr-sectors': {
    kind: 'numeric',
    id: 'ch17/wrap/q-csr-sectors',
    prompt: (
      <>
        Uniform rows of {LEN} floats in CSR, the array starting on a 32-byte boundary. A warp of {WARP} threads runs its{' '}
        {LEN} iterations; in iteration <code>t</code>, thread <code>l</code> reads <code>value[{LEN}*l + t]</code>.
        Counted per instruction, each iteration touches {csrIter(0).transactions} sectors, {SECTORS.csrRequests} sector
        requests in all. How many <em>distinct</em> 32-byte sectors of <code>value</code> do those {LEN} iterations touch?
      </>
    ),
    answer: SECTORS.csrDistinct,
    unit: 'sectors',
    explain: (
      <p>
        Each thread's row is {LEN * 4} bytes, two sectors, and the thread reads every float in both: {WARP} × 2 ={' '}
        {SECTORS.csrDistinct}. ELL on the same rows touches {SECTORS.ellPerIter} sectors per iteration,{' '}
        {SECTORS.ellDistinct} in all. Counted per instruction CSR asks for {SECTORS.csrRequests / SECTORS.ellDistinct}{' '}
        times as much; the bytes that have to come from memory are the same. If a cache keeps each thread's two sectors
        between its iterations, most of the gap closes. Neither the book nor the vendor's coalescing rule says how much a
        cache keeps, which is why lab 17.1 asks you to predict CSR against ELL on uniform rows and then measure.
      </p>
    ),
    source: 'ours, with the 32-byte sector of ch06-facts A2; ch. 17 §17.3–17.4, pp. 409–413',
  },
  'ch17/wrap/q-ragged': {
    kind: 'numeric',
    id: 'ch17/wrap/q-ragged',
    prompt: (
      <>
        A decode batch holds {RAGGED.seqs} sequences: {RAGGED.seqs - 1} with {RAGGED.short} cached tokens and one with{' '}
        {n(RAGGED.long)}. An attention kernel pads every sequence's cache to the longest and walks every padded slot. What
        percentage of the slots it walks hold a real token?
      </>
    ),
    answer: Math.round(RAGGED_PCT * 100) / 100,
    tol: 0.02,
    unit: '%',
    explain: (
      <p>
        {RAGGED.seqs - 1} × {RAGGED.short} + {n(RAGGED.long)} = {n(RAGGED_REAL)} real tokens in {RAGGED.seqs} ×{' '}
        {n(RAGGED.long)} = {n(RAGGED_SLOTS)} slots: {RAGGED_PCT.toFixed(1)}%. This is ELL with one long row, the book's
        1000 × 1000 example in other clothes, and the chapter's fixes carry over. Stop each sequence at its own length
        (<code>nnzPerRow</code>: the walking shrinks, the storage does not). Store the sequences back to back with offsets
        (<code>rowPtrs</code>). Group sequences of similar length (the JDS sort, with an index to put results back). Or
        send the long one down its own path (the hybrid).
      </p>
    ),
    source: 'ours, after ch. 17 §17.4–17.6, pp. 412–417',
  },
  'ch17/wrap/q-moe-combine': {
    kind: 'choice',
    id: 'ch17/wrap/q-moe-combine',
    prompt: (
      <>
        A mixture-of-experts layer sends each token to its top 2 experts. To combine, one thread per (token, expert) result
        multiplies the expert's output row by the router weight and atomically adds it, element by element, into the
        token's hidden state, which already holds the residual. Two runs on the same input give logits that differ in the
        last bits. Which change makes the combine reproducible bit for bit?
      </>
    ),
    options: [
      {
        label:
          "Give each token's row to one thread or block, which gathers its two weighted expert results through the inverse permutation and adds them in a fixed order, with no atomic.",
        correct: true,
      },
      {
        label: (
          <>
            Keep the atomic, with <code>memory_order_seq_cst</code> instead of relaxed.
          </>
        ),
        why: 'A memory order constrains how other memory operations are seen around the add. The two adds into one element still land one at a time in an undefined order, and (r + a) + b and (r + b) + a can round differently.',
      },
      {
        label: 'Sort the (token, expert) pairs by token before the combine.',
        why: "That orders the threads' inputs, not the moment each atomic lands. The adds still serialise in whatever order the hardware runs them.",
      },
      {
        label: 'Accumulate in double precision with double atomics.',
        why: 'The rounding error shrinks, but two orders can still round differently in double, and the conversion back can carry that into the last bit. Rarer is not reproducible.',
      },
    ],
    explain: (
      <p>
        This is COO against CSR again. One thread per (token, expert) pair is COO: several threads target one output, so
        the add must be atomic, and atomics on one address serialise in an undefined order, so the float rounding depends
        on the run. One owner per output row is CSR: the code fixes the order of its additions. The residual matters: with
        top 2 and an output starting at zero, <code>0 + a + b</code> and <code>0 + b + a</code> are equal, since float
        addition commutes; with a third term in the sum they need not be. The vendor libraries offer the same trade for
        SpMV: an atomic COO algorithm that is faster, and a deterministic one.
      </p>
    ),
    source: 'ch. 17 §17.2, pp. 406–407; ch09-facts A3; ch10-facts A9; ch17-facts A2; the combine kernel is ours',
  },
};

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = realBackends(reports);
  const lab = 'ch17/lab1_spmv';
  const passing = [...backends, 'emu'].filter((b) => labStatus(reports[b], lab, 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch17/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul data-list="carry">
        <li>
          The storage format decides which thread owns which piece of work, and that decides whether you need atomics. One
          thread per non-zero collides on the output; one thread per row owns it. Atomics cost throughput and fix no
          order, so float results can change from run to run.
        </li>
        <li>
          Whether loads coalesce is a property of the index neighbouring threads compute. CSR's{' '}
          <code>rowPtrs[r] + t</code> scatters them; ELL's <code>t·stride + r</code> lines them up. Padding plus a
          column-major layout buys that, at a price the longest row sets.
        </li>
        <li>
          Trip counts that come from data make warps diverge, and ELL alone does not fix it. Spilling the long tails
          (hybrid), sorting rows by length (JDS, sliced ELL) or bucketing them does, each at a preprocessing cost and,
          when rows move, with an index array to put the results back.
        </li>
        <li>
          Preprocessing pays only over reuse: many solver iterations with one matrix, or weights repacked once at load
          time before many forward passes.
        </li>
        <li>
          SpMV accumulates, <code>y = A·x + y</code>. Write <code>+=</code> in every kernel and mark pads with column −1
          and a test, so that formats compose in either order and an infinite <code>x</code> cannot leak into rows that
          do not use it.
        </li>
        <li>
          Performance depends on the data. The best format follows where the non-zeros fall, the request-counting models
          ignore caches, and only a measurement on your device says how much each move is worth.
        </li>
      </ul>
      <Source section="17.8" pages="421–422" />

      <h2>Exercise 5, answered</h2>
      <p>
        The book leaves the JDS kernel as an exercise. One thread per sorted row, rows longest first. In iteration{' '}
        <code>t</code>, <code>iterPtr[t+1] − iterPtr[t]</code> rows still have an element, and because the rows are sorted
        they are the first ones, so thread <code>r</code> takes part while <code>r</code> is below that count. Its element
        is at <code>iterPtr[t] + r</code>: consecutive threads read consecutive slots, as in ELL, with no padding. On Fig.
        17.14, <code>iterPtr = {list(J14.iterPtr)}</code>, so the threads at work in iterations 0 to {PER_ITER.length - 1}{' '}
        number {PER_ITER.slice(0, -1).join(', ')} and {PER_ITER[PER_ITER.length - 1]}. Page{' '}
        <a href={href('ch17', '17.5')}>17.5</a> asks for the loop test as a question.
      </p>
      <Source section="17.6, 17.9" pages="416–417, 422" figs="Figs. 17.14, 17.15" />
      <JdsKernel />

      <h2>A mixed check</h2>
      <p>
        These draw on the whole chapter and on situations the pages did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly on the first recorded try.` : ''}
      </p>
      <Beyond>
        <p>
          The defect in the kernel, the transpose, the cache count, the padded batch and the expert combine are ours. They
          rest on the book's pages and on the facts each answer names.
        </p>
      </Beyond>
      <Ask q={q['ch17/wrap/q-jds-unpermuted']} label="Find the defect" />
      <Ask q={q['ch17/wrap/q-transpose']} />
      <Ask q={q['ch17/wrap/q-csr-sectors']} label="Work it out" />
      <Ask q={q['ch17/wrap/q-ragged']} label="Work it out" />
      <Ask q={q['ch17/wrap/q-moe-combine']} />

      <h2>The book's exercises</h2>
      <p>
        The chapter ends with five exercises (p. 422). The book prints no answers, so the answers in this course are ours.
      </p>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto', marginBottom: '1.2rem' }}>
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
              <td>A 4 × 4 matrix in COO, CSR, ELL and JDS</td>
              <td>
                the format explorer on <a href={href('ch17', '17.2')}>17.2</a> with the "Exercise 1" matrix loaded (every
                tab), and its pad question. Its JDS is <code>row = {list(J_EX1.row)}</code>,{' '}
                <code>iterPtr = {list(J_EX1.iterPtr)}</code>
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td>How many integers each format needs, from m, n and z</td>
              <td>
                the explorer's "Exercise 2's count" line and the question on what is missing, on{' '}
                <a href={href('ch17', '17.2')}>17.2</a>
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td>COO to CSR with a histogram and a prefix sum</td>
              <td>
                the scan question on the <a href={href('ch17', 'start')}>start page</a>; the animation "Routing tokens to
                experts" and its offset question on <a href={href('ch17', '17.5')}>17.5</a>; the sketch below
              </td>
            </tr>
            <tr>
              <td>4</td>
              <td>Host code for the hybrid ELL-COO format</td>
              <td>
                the hybrid pass of <a href={href('ch17', 'labs')}>lab 17.1</a>, with the COO part on the device; the order
                question on <a href={href('ch17', '17.2')}>17.2</a>
              </td>
            </tr>
            <tr>
              <td>5</td>
              <td>A JDS kernel</td>
              <td>
                the loop-test question on <a href={href('ch17', '17.5')}>17.5</a>, and the kernel above
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        Exercise 3 in three passes. Count the entries of each row with a histogram over <code>rowIdx</code> (chapter 9).
        Take the exclusive scan of the counts and append the total: that is <code>rowPtrs</code> (chapter 11). Then scatter
        each entry to <code>rowPtrs[row]</code> plus its rank among the earlier entries of the same row, copying{' '}
        <code>colIdx</code> and <code>value</code>. An atomic cursor per row gives that rank in an undefined order, which
        the CSR kernel tolerates (p. 408); a rank computed from the input order keeps every row's entries in the order they
        arrived. Exercise 4 asks for the COO part on the host. If the ELL kernel assigns, as Fig. 17.12 does, that host pass
        has to come after it; with <code>+=</code> either order works.
      </p>
      <Source section="17.3, 17.9" pages="408, 410, 422" />
      <p>
        The vendor libraries do the conversion in one call, and CUB's own SpMV is gone, with the advice to use cuSPARSE.
      </p>
      <Today checked="2026-11-20 (notes/today/ch17-facts.md A9)">
        <p>
          <code>cusparseXcoo2csr</code> and <code>rocsparse_coo2csr</code>; <code>cub::DeviceSpmv</code> was removed in
          CCCL 3.0.
        </p>
      </Today>

      <h2>Where the PDF looks wrong</h2>
      <p>
        The copy of the book this course was built from has a few apparent slips in chapter 17. They are listed so that you
        are not thrown by them when you read the book alongside. The first and the last change code.
      </p>
      <ul data-list="errata">
        <li>
          Fig. 17.12 ends with <code>y[row] = sum</code>, where Fig. 17.9 has <code>y[row] += sum</code> and the text calls
          both an accumulation (pp. 409, 412). SpMV accumulates; every kernel in this course writes <code>+=</code>, and the
          order question on <a href={href('ch17', '17.2')}>17.2</a> shows what the assignment breaks.
        </li>
        <li>p. 413 says the ELL index "was calculated in Fig. 17.9"; it is in Fig. 17.12.</li>
        <li>
          p. 414 says three non-zeros of "row 2" move to COO. Fig. 17.13 and the counts (width 5 to 2, 22 pads to 3) say
          row 1.
        </li>
        <li>
          The 1000 × 1000 example (pp. 412–413) gives the other rows "fewer than 10" non-zeros, which cannot add up to 1%
          beside a row of 200. Page 17.5 keeps the 1%; the ratio is 19.05, which the book rounds to 20.
        </li>
        <li>
          p. 416 sorts the rows "in increasing order"; the paragraph before it and Fig. 17.14 sort longest first, which this
          course follows.
        </li>
        <li>p. 417 opens the evaluation of JDS with "Now we examine the ELL format".</li>
        <li>
          p. 417 speaks of reordering the solution's <code>x</code> elements. Sorting rows reorders equations, so it is{' '}
          <code>y</code> that goes back through the <code>row</code> array (our reading, and the first question above).
        </li>
        <li>Fig. 17.18's caption reads "A parallel SpMV/CSR kernel"; the listing is the CSC kernel.</li>
        <li>
          Fig. 17.18 reads <code>rowIdxs</code> and <code>values</code>, where the text and Fig. 17.16 say{' '}
          <code>rowIdx</code> and <code>value</code>. The copy on 17.2 uses the singular names.
        </li>
        <li>
          p. 410's "the disadvantages of CSR over COO" means CSR's disadvantages compared with COO. Not a slip, but easy to
          misread.
        </li>
        <li>
          The book never says what a pad slot's column holds, though a loop over the full width reads it (p. 411). Column
          0 gives NaN in every padded row when <code>x[0]</code> is infinite; this course uses −1 and a test, as the
          libraries do.
        </li>
      </ul>
      <p>And what the listings leave to you, which a real build needs:</p>
      <ul data-list="gaps">
        <li>
          No <code>COOMatrix</code>, <code>CSRMatrix</code>, <code>ELLMatrix</code> or <code>CSCMatrix</code> struct, and
          no host or launch code. The snippets on 17.2 define the structs; the lab passes raw pointers.
        </li>
        <li>
          No <code>#include &lt;cuda/atomic&gt;</code> for the <code>cuda::atomic_ref</code> of Figs. 17.5 and 17.18.
        </li>
        <li>
          The alignment unit "such as 64-bytes" (p. 413) is in no NVIDIA document read for this course; today's unit is
          the 32-byte sector, a stride that is a multiple of 8 floats.
        </li>
        <li>No JDS kernel: exercise 5. The one above is ours.</li>
      </ul>
      <Aside>
        <p>
          These are our readings of one PDF (notes/ch17.md, "Unclear in the PDF"), with the vendor points checked on
          2026-11-20 (notes/today/ch17-facts.md A1, A3, A6, A10, E1). If your copy differs, trust your copy.
        </p>
      </Aside>

      <h2>What this chapter left out</h2>
      <p>
        The chapter is condensed in this course, so here is what is not covered and where to find it. The iterative-solver
        background, Conjugate Gradient, positive-definite matrices and fill-ins (pp. 402–403), gets one sentence on the
        start page. The history behind the names ELLPACK and JDS (pp. 410, 416) is not here. The sectioned JDS variant (p.
        417) has no kernel: it appears as sliced ELL in the explorer on 17.5 and as cuSPARSE's Sliced ELL. The CSC kernel
        (Figs. 17.17 and 17.18, pp. 419–420) is shown corrected and scored on 17.2 but never run in a lab, and
        vector–matrix products and SpMSpV (p. 421) get one paragraph there. You never write the COO kernel (Fig. 17.5): it
        is a snippet on 17.2 and a provided kernel in the lab's hybrid pass.
      </p>
      <p>
        Exercise 3 is worked as an animation and questions, not coded, and exercise 4's host loop is replaced by the lab's
        device pass. Bell and Garland's paper (p. 421, reference 3) is named only, because its device figures are from
        2009 hardware. The book's claims that real matrices have tens to hundreds of non-zeros per row (p. 409) and that
        sparse code reaches far fewer FLOPS than dense (p. 422) are presented as the book's, which no vendor page read for
        this course states; the lab measures your own kernels instead.
      </p>
      <Today checked="2026-11-20 (notes/today/ch17-facts.md A12, A13)">
        <p>Both generalisations: could not verify. Bell and Garland's paper is still hosted by NVIDIA Research.</p>
      </Today>
      <p>Nothing later in the course depends on what was left out.</p>

      <h2>The project so far</h2>
      <p>
        This chapter adds no milestone to the through-line project. A dense transformer's inference step multiplies dense
        matrices, and nothing on its main path runs an SpMV. What the chapter teaches returns as patterns: block-sparse
        attention masks stored like ELL, and mixture-of-experts routing as a COO to CSR conversion. Lab 17.1 is optional.{' '}
        <span data-lab171={passing.join(',') || 'none'}>
          {backends.length === 0 && !reports['emu']
            ? 'No lab results are recorded on this machine yet.'
            : passing.length
              ? `Your own version passes on ${passing.map(where).join(' and ')} on this machine.`
              : 'Your own version has not passed on this machine yet.'}
        </span>{' '}
        See the <a href={href('ch17', 'labs')}>lab</a>.
      </p>
      <p>
        Chapter 18 stores graphs in the same three formats: CSR for a vertex's outgoing edges, CSC for its incoming ones,
        COO for an edge list. Its breadth-first search kernels look like this chapter's SpMV kernels, and one level of the
        search can be written as an SpMV plus a few vector operations. What it adds is work whose amount and place change while the kernel runs. Start
        at <a href={href('ch18', 'start')}>chapter 18</a>.
      </p>
      <Source section="18.1, 18.4" pages="427, 438" />
    </>
  );
}
