import { Ask, Predict, type Question } from '../../components/Ask';
import { Beyond, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { fmt } from '../../lib/reports';
import { warpAccess } from '../../sim/coalesce';
import { FIG171, countByRow, exclusiveScan, fig171Dense, float32Sums } from '../../sim/sparseroute';
import { breakEvenDensity, bytesPerPosition, warpStats } from '../../sim/sparsescale';

// The start page of chapter 17: what SpMV computes, what the course keeps of the chapter,
// the five criteria, a question about bytes before any format, a check on chapters 4, 6,
// 9 and 11 and appendix A, and a map. Every number below comes from src/sim/sparseroute.ts,
// src/sim/sparsescale.ts or src/sim/coalesce.ts.

const DENSE = fig171Dense();
const LENGTHS = countByRow(FIG171.rowIdx, FIG171.rows); // [2, 3, 2, 1]
const ROWPTRS = exclusiveScan(LENGTHS); // [0, 2, 5, 7, 8]
const LOOP = warpStats(LENGTHS, LENGTHS.length); // 8 useful of 12
const LOOP_PCT = Math.round(LOOP.efficiency * 100); // 67
const lanes = (stride: number) => Array.from({ length: 32 }, (_, t) => stride * t * 4);
const STRIDE16 = warpAccess(lanes(16), 4, 32).transactions; // 32
const UNIT = warpAccess(lanes(1), 4, 32).transactions; // 4
const ORDER = float32Sums([1e8, 1, -1e8]); // [0, 1]
const SPACING = Math.pow(2, Math.floor(Math.log2(1e8)) - 23); // float32 spacing at 1e8: 8

// The hook: fp16 pruned to 10% in CSR (fp16 values, int32 indices) against dense fp16 and dense 4-bit.
const HOOK_COLS = 4096;
const PRUNED = bytesPerPosition(0.1, 2, 4, Infinity); // 0.6
const PTR_SHARE = bytesPerPosition(0.1, 2, 4, HOOK_COLS) - PRUNED; // 4 / 4096
const FEWER = 2 / PRUNED; // 3.3
const BREAK_EVEN = breakEvenDensity(0.5, 2, 2) * 100; // 12.5
const HALF = bytesPerPosition(0.5, 2, 2, Infinity); // 2 bytes at half the weights kept
const list = (xs: readonly number[]) => '[' + xs.join(', ') + ']';

export const questionsS170: Record<string, Question> = {
  'ch17/start/q-breakeven': {
    kind: 'numeric',
    id: 'ch17/start/q-breakeven',
    prompt: (
      <>
        A dense weight matrix quantised to 4 bits reads 0.5 bytes per weight. Store the same matrix pruned instead, in
        CSR with 16-bit values and 16-bit column indices, and leave out the row pointers. Below what density, in percent
        of weights kept, does CSR read fewer bytes of matrix than the 4-bit dense one?
      </>
    ),
    answer: BREAK_EVEN,
    tol: 0.005,
    unit: '%',
    explain: (
      <p>
        Every kept weight costs 2 + 2 = 4 bytes, so CSR reads 4·d bytes per position against 0.5: d = 0.5 / 4 ={' '}
        {fmt(BREAK_EVEN)}%. Pruning that keeps half the weights reads {fmt(HALF)} bytes per position, four times the
        quantised matrix. With quantised dense weights, pruning has to be extreme before an index format wins on bytes,
        and bytes are the part of the argument that favours sparse: page <a href={href('ch17', '17.5')}>17.5</a> adds
        what the book says about tiling, and its explorer lets you change the bytes per value.
      </p>
    ),
    source: 'ours, after ch. 17 §17.8, p. 422',
  },
  'ch17/start/q-warp-loop': {
    kind: 'numeric',
    id: 'ch17/start/q-warp-loop',
    prompt: (
      <>
        Four threads, one per row of the matrix above, each loop over their row's non-zeros: trip counts{' '}
        {LENGTHS.join(', ')}. They run as one group that keeps issuing iterations until its longest thread is done, and
        a lane whose thread has finished sits idle. What percentage of the group's lane-iterations do useful work?
      </>
    ),
    answer: LOOP_PCT,
    tol: 0.01,
    unit: '%',
    explain: (
      <p>
        The group runs {LOOP.longest} iterations, its longest trip, so {LENGTHS.length} × {LOOP.longest} ={' '}
        {LENGTHS.length * LOOP.longest} lane-iterations, of which {LOOP.useful} work: {fmt(LOOP.efficiency * 100)}%. This
        is chapter 4's loop with trip counts that come from data (Fig. 4.10). In this chapter it is the load-balance
        problem of the CSR kernel, which gives each thread a row and lets the row decide how long the thread runs (page{' '}
        <a href={href('ch17', '17.2')}>17.2</a>). Sorting rows by length is the fix on page{' '}
        <a href={href('ch17', '17.5')}>17.5</a>.
      </p>
    ),
    source: 'ch. 4 §4.5, p. 81 (Fig. 4.10); ch. 17 Fig. 17.1',
  },
  'ch17/start/q-stride16': {
    kind: 'numeric',
    id: 'ch17/start/q-stride16',
    prompt: (
      <>
        On an NVIDIA GPU a warp's load is served in 32-byte segments. The 32 lanes of a warp each load a 4-byte float,
        lane <code>t</code> the element at index <code>16*t</code> of an array that starts on a segment boundary. How many
        segments does this one load touch?
      </>
    ),
    answer: STRIDE16,
    unit: 'segments',
    explain: (
      <p>
        Lane <code>t</code>'s float starts 64 bytes after lane <code>t − 1</code>'s, so no two lanes share a segment:{' '}
        {STRIDE16} segments, each delivering 4 useful bytes of its 32. Index <code>t</code> would touch {UNIT}. This is the
        CSR kernel's first iteration when every row has 16 non-zeros: thread <code>t</code> reads{' '}
        <code>value[16*t]</code> (Fig. 17.8). ELL's column-major layout turns the same reads into index <code>t</code>{' '}
        (page <a href={href('ch17', '17.2')}>17.2</a>).
      </p>
    ),
    source: "ch. 6 §6.1, pp. 124–132; NVIDIA's 32-byte unit: ch06-facts A2",
  },
  'ch17/start/q-lost-update': {
    kind: 'choice',
    id: 'ch17/start/q-lost-update',
    prompt: (
      <>
        <code>y[0]</code> starts at 0. Two threads each execute a plain <code>y[0] += v</code>, one with{' '}
        <code>v = 1</code> and one with <code>v = 7</code>, with no atomic and no barrier. Which final values of{' '}
        <code>y[0]</code> are possible?
      </>
    ),
    options: [
      { label: '1, 7 or 8.', correct: true },
      { label: 'Only 8: each += is one instruction.', why: 'It is a load, an add and a store. The other thread can run between them.' },
      { label: '8, or 0 when the two collide.', why: 'Each thread stores a sum it computed, 1, 7 or 8. A collision loses one of them, not both.' },
      { label: 'Any value: a race is undefined.', why: 'Each store writes a whole float the thread computed. Nothing produces values other than 1, 7 and 8.' },
    ],
    explain: (
      <p>
        Each thread reads <code>y[0]</code>, adds, and writes back. If both read 0 before either writes, the later write
        wins and leaves 1 or 7; if one finishes before the other reads, 8. These are row 0's two non-zeros of the matrix
        above, multiplied by an <code>x</code> of ones. One thread per non-zero (COO) puts both on <code>y[0]</code>, so
        the book's COO kernel adds with an atomic (Fig. 17.5); one thread per row (CSR) owns <code>y[0]</code> and needs
        none (page <a href={href('ch17', '17.2')}>17.2</a>).
      </p>
    ),
    source: 'ch. 9 §9.2, pp. 203–209; ch. 17 §17.2, pp. 404–407',
  },
  'ch17/start/q-order': {
    kind: 'choice',
    id: 'ch17/start/q-order',
    prompt: (
      <>
        One row's three products are 1e8, 1 and −1e8. Three threads add them into <code>y = 0</code> with float32 atomic
        adds, in whatever order the hardware serialises them. Which results can appear?
      </>
    ),
    options: [
      { label: 'Always 1: atomics make the sum exact.', why: 'An atomic makes each add indivisible. It neither fixes the order nor removes rounding.' },
      { label: '0 or 1, depending on the order.', correct: true },
      { label: 'Always 0: float32 cannot hold 1e8 + 1.', why: 'The 1 survives when the two large terms cancel first: (1e8 − 1e8) + 1 = 1.' },
      { label: '0, 1 or 2, as rounding goes up or down.', why: 'Round to nearest takes 1e8 + 1 back to 1e8 every time. No order produces 2.' },
    ],
    explain: (
      <p>
        Near 1e8, consecutive float32 values are {SPACING} apart, so 1e8 + 1 rounds back to 1e8. Orders that add the 1
        to a large term first end at 0; orders that cancel the two large terms first end at 1. The model, trying every
        order: {list(ORDER)}. Atomics on one address run one at a time in an undefined order (chapter 9), so either result
        can appear on any run. This is the book's second objection to atomics in SpMV (p. 407). NVIDIA's and AMD's
        libraries both offer a slower, deterministic COO algorithm for this reason (page{' '}
        <a href={href('ch17', '17.5')}>17.5</a> lists them).
      </p>
    ),
    source: 'appendix A §A.6, pp. 593–594; ch. 17 p. 407; ch17-facts A2',
  },
  'ch17/start/q-scan': {
    kind: 'numeric',
    id: 'ch17/start/q-scan',
    prompt: (
      <>
        The rows of the matrix above hold {LENGTHS.join(', ')} non-zeros. Take the exclusive scan of {list(LENGTHS)}. What
        is its element 2?
      </>
    ),
    answer: ROWPTRS[2],
    explain: (
      <p>
        The exclusive scan is {list(ROWPTRS.slice(0, -1))}: each element is the sum of the counts before it. Append the
        total, {ROWPTRS[ROWPTRS.length - 1]}, and you have {list(ROWPTRS)}, which is exactly CSR's{' '}
        <code>rowPtrs</code> (Fig. 17.7): row 2 starts at {ROWPTRS[2]}. Exercise 3 builds CSR from COO this way: count per
        row, scan, scatter. Page <a href={href('ch17', '17.5')}>17.5</a> runs the same three passes to group tokens by
        expert.
      </p>
    ),
    source: 'ch. 11 §11.1, pp. 252–254; ch. 17 Fig. 17.7 and exercise 3, p. 422',
  },
};

// ---------------------------------------------------------------- Fig. 17.1, redrawn
const C = 36;
const AX = 74;
const AY = 34;
const VX = AX + 4 * C + 34; // x
const YX = VX + C + 34; // y
const OUTX = YX + C + 48; // result y

function Vector({ x, name, label }: { x: number; name: string; label: string }) {
  return (
    <g data-vector={name}>
      <text x={x + C / 2 - 1} y={AY - 10} fontSize="13" fontWeight="600" textAnchor="middle" style={{ fill: 'var(--global)' }}>
        {label}
      </text>
      {[0, 1, 2, 3].map((i) => (
        <g key={i}>
          <rect x={x} y={AY + i * C} width={C - 3} height={C - 3} rx="2" fill="var(--global-wash)" stroke="var(--global)" />
          <text x={x + (C - 3) / 2} y={AY + i * C + 22} fontSize="12" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
            {name}
            {i}
          </text>
        </g>
      ))}
    </g>
  );
}

function Fig171() {
  return (
    <figure style={{ margin: '0.4rem 0 1.4rem' }}>
      <svg
        viewBox={`0 0 ${OUTX + C + 6} ${AY + 4 * C + 14}`}
        role="img"
        aria-label={`The book's 4 by 4 example: rows ${DENSE.map((r) => '[' + r.join(' ') + ']').join(', ')}, multiplied by x and added to y.`}
        style={{ width: '100%', maxWidth: OUTX + C + 6, display: 'block', fontFamily: 'var(--sans)' }}
      >
        <text x={AX + 2 * C} y={AY - 10} fontSize="13" fontWeight="600" textAnchor="middle" style={{ fill: 'var(--global)' }}>
          A
        </text>
        {DENSE.map((row, r) => (
          <g key={r} data-row={r}>
            <text x={AX - 10} y={AY + r * C + 22} fontSize="12" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
              row {r}
            </text>
            {row.map((v, c) => (
              <g key={c} data-cell={`${r},${c}`} data-value={v}>
                <rect
                  x={AX + c * C}
                  y={AY + r * C}
                  width={C - 3}
                  height={C - 3}
                  rx="2"
                  fill={v ? 'var(--global-wash)' : 'var(--cell)'}
                  stroke={v ? 'var(--global)' : 'var(--rule)'}
                />
                {v ? (
                  <text x={AX + c * C + (C - 3) / 2} y={AY + r * C + 22} fontSize="14" textAnchor="middle" style={{ fill: 'var(--ink)' }}>
                    {v}
                  </text>
                ) : null}
              </g>
            ))}
          </g>
        ))}
        <text x={VX - 17} y={AY + 2 * C + 4} fontSize="18" textAnchor="middle" style={{ fill: 'var(--ink)' }}>×</text>
        <Vector x={VX} name="x" label="x" />
        <text x={YX - 17} y={AY + 2 * C + 4} fontSize="18" textAnchor="middle" style={{ fill: 'var(--ink)' }}>+</text>
        <Vector x={YX} name="y" label="y" />
        <text x={OUTX - 24} y={AY + 2 * C + 4} fontSize="18" textAnchor="middle" style={{ fill: 'var(--ink)' }}>→</text>
        <Vector x={OUTX} name="y" label="new y" />
      </svg>
      <figcaption className="muted" style={{ fontFamily: 'var(--sans)', fontSize: '0.8rem' }}>
        The book's running example (Fig. 17.1), redrawn as the whole operation <code>y = A·x + y</code>. The book colours
        each non-zero; here rows are labelled instead, because colour in this course means where data lives. Every array
        is in the global-memory colour: the chapter's kernels read them all from there.
      </figcaption>
    </figure>
  );
}

const CRITERIA: { name: string; asks: string; side: string }[] = [
  { name: 'Space efficiency', asks: 'How much memory the matrix takes.', side: 'compaction' },
  { name: 'Flexibility', asks: 'How easily a non-zero can be added or removed.', side: 'neither' },
  { name: 'Accessibility', asks: 'Which questions are cheap: all non-zeros of a given row; the row and column of a given non-zero.', side: 'neither' },
  { name: 'Memory-access efficiency', asks: "Whether the kernel's loads coalesce.", side: 'regularity' },
  { name: 'Load balance', asks: 'Whether threads get equal work, so that warps do not diverge.', side: 'regularity' },
];

export function S170() {
  return (
    <>
      <p>
        This chapter is about one computation. A matrix <i>A</i> that is mostly zeros multiplies a dense vector{' '}
        <i>x</i>, and the product is added to a dense vector <i>y</i>: <code>y = A·x + y</code>. The book calls it SpMV,
        sparse matrix–vector multiplication and accumulation, and the accumulate matters: its kernels add into{' '}
        <code>y</code> rather than overwrite it.
      </p>
      <Source section="17.1" pages="402–403" />
      <Fig171 />
      <p>
        The book's reason for a whole chapter is iterative solvers for large linear systems. Each iteration evaluates{' '}
        <code>A·x + y</code> with a new guess for <code>x</code> and the same <code>A</code> as every other iteration.
        Keep that repetition in mind: it is what makes it affordable to convert or sort <code>A</code> once before the
        first product.
      </p>
      <Source section="17.1" pages="402–403" />
      <p>
        Today's libraries keep the accumulate and add two scalars: they compute <code>y = α·op(A)·x + β·y</code>, where{' '}
        <code>op(A)</code> is <code>A</code> or its transpose. The book's SpMV is α = 1, β = 1; a plain product is β = 0.
      </p>
      <Today checked="2026-11-20 (ch17-facts A4, A10)">
        <p>
          <code>cusparseSpMV</code> (cuSPARSE 13.5), <code>rocsparse_spmv</code> and <code>hipsparseSpMV</code> all use
          this formula. The standard's routine, Sparse BLAS <code>usmv</code>, has <code>y</code> in-out and no β.
        </p>
      </Today>

      <h2>What this course keeps</h2>
      <p>
        Dense LLM inference rarely runs an SpMV. Sparse patterns do turn up around it: attention masks that skip blocks,
        mixture-of-experts layers that send each token to a few experts, pruned weights. What carries over from the
        chapter is less its catalogue of formats than six techniques you will reuse in kernels that have nothing to do
        with linear solvers:
      </p>
      <Beyond>
        <p>
          The LLM framing is ours. The code behind it, in PyTorch, llama.cpp and MLX, is in ch17-facts D1 to D7, and page
          17.5 walks through one case.
        </p>
      </Beyond>
      <ul>
        <li>
          <b>The thread-to-work mapping decides whether you need atomics.</b> One thread per non-zero collides on the
          output; one thread per row owns it. <span className="muted">(pp. 405–409)</span>
        </li>
        <li>
          <b>Trip counts that come from data cause divergence.</b> When each thread loops over its own row, the threads of
          a warp finish at different times. <span className="muted">(p. 410)</span>
        </li>
        <li>
          <b>Padding plus a transposed layout buys coalescing</b>, at a price set by the longest row.{' '}
          <span className="muted">(pp. 410–413)</span>
        </li>
        <li>
          <b>Split a regular bulk from an irregular tail</b>, and run each with the kernel that suits it.{' '}
          <span className="muted">(pp. 414–415)</span>
        </li>
        <li>
          <b>Sort or bucket by length, and keep the inverse permutation</b> to put the results back.{' '}
          <span className="muted">(pp. 416–417)</span>
        </li>
        <li>
          <b>Amortise preprocessing over reuse</b>: a conversion that costs several products pays when the same matrix is
          used many times. <span className="muted">(pp. 414–415, 417)</span>
        </li>
      </ul>
      <p>
        What is left out: the solver background (Conjugate Gradient, positive-definite matrices, and the fill-ins that
        make inverting a large sparse matrix impractical) and the history behind the names ELL and JDS. None of it changes
        how a kernel is written.
      </p>
      <Source section="17.1" pages="402" />

      <h2>Compaction against regularity</h2>
      <p>
        Every sparse format drops zeros, which the book calls compaction, and pays for it with irregularity: the data no
        longer has a shape a thread can compute its way around. On a GPU, irregularity has three costs: memory bandwidth
        left unused, control divergence, and load imbalance. Formats sit at different points between compacting hard and
        staying regular, and which one is fastest depends on where the non-zeros fall.
      </p>
      <Source pages="401" />
      <p>
        The book scores every format against the same five criteria. They are the columns of the format explorer on page{' '}
        <a href={href('ch17', '17.2')}>17.2</a>, and they are worth keeping as questions to ask of any data layout you
        design for a kernel.
      </p>
      <Source section="17.1" pages="403–404" />
      <table className="data" data-criteria>
        <thead>
          <tr>
            <th>Criterion</th>
            <th>What it asks</th>
            <th>Side of the trade</th>
          </tr>
        </thead>
        <tbody>
          {CRITERIA.map((c) => (
            <tr key={c.name}>
              <td>{c.name}</td>
              <td>{c.asks}</td>
              <td>{c.side}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>A question before any format</h2>
      <p>
        A matrix-vector product reads every weight once. Pruning drops weights; quantising shrinks them. Both cut the bytes
        read, and you already know quantisation. Compare the two on bytes alone before you meet a single sparse format.
      </p>
      <Predict
        id="ch17/start/p-bytes"
        prompt={
          <>
            A weight matrix of fp16 values is pruned to keep 10% of its weights. The kept weights are stored in CSR: an
            fp16 value and a 32-bit column index each, plus one 32-bit row pointer per row. How many bytes of matrix does
            a matrix-vector product read per position of the original matrix? Dense fp16 reads 2.
          </>
        }
        unit="bytes"
      >
        {(v) => (
          <div data-reveal="ch17/start/p-bytes">
            <p>
              The model says <b data-out="pruned">{fmt(PRUNED)}</b>: 0.1 × (2 + 4). The row pointer adds 4 bytes per row,
              or {fmt(PTR_SHARE)} per position on a row of {HOOK_COLS} columns, which is noise. That is{' '}
              <b data-out="fewer">{fmt(FEWER, 2)}</b> times fewer bytes than dense fp16.{' '}
              {Math.abs(v - PRUNED) < 0.05 ? 'You had it.' : 'Every kept weight pays for its index as well as its value.'}
            </p>
            <p>
              Now quantise instead of pruning: a dense matrix of 4-bit weights reads <b>0.5</b> bytes per position and no
              indices at all. Pruning to 10% loses to it, because every kept weight drags a 4-byte index along. Even with
              16-bit indices, the question below shows how sparse a matrix has to be before CSR wins.
            </p>
          </div>
        )}
      </Predict>
      <Ask q={questionsS170['ch17/start/q-breakeven']} label="Before any format" />
      <p>
        The book ends the chapter with the same warning from the other side. A dense matrix has implicit indices and is
        easy to tile for the memory hierarchy; a sparse format stores explicit indices that must be read, and is hard to
        tile. So a sparse format pays off only when the matrix is very sparse. The book's rule of thumb for real
        matrices, tens to hundreds of non-zeros in rows that are thousands to millions long, is a density far below what
        pruning a model leaves.
      </p>
      <Source section="17.3, 17.8" pages="409, 422" />
      <p>
        PyTorch's documentation says the same: sparse formats "are not always advantageous", and trying
        one you "might find your execution time to increase rather than decrease".
      </p>
      <Today checked="2026-11-20 (ch17-facts D1)">
        <p>PyTorch 2.15's sparse page, which also calls its semi-structured (2:4) sparsity a prototype.</p>
      </Today>
      <p>
        Pruned and quantised weights are where this question meets your work. When a sparse format does win on bytes, the
        rest of the chapter is about whether a kernel can turn those bytes into speed.
      </p>
      <Beyond>
        <p>The bridge to pruned and quantised weights, and the arithmetic of the hook, are ours.</p>
      </Beyond>

      <h2>Before you start</h2>
      <p>
        Five questions on what this chapter takes from chapters 4, 6, 9 and 11 and appendix A, in the order it uses them.
        Each one comes back in <a href={href('review')}>review</a>.
      </p>
      <Ask q={questionsS170['ch17/start/q-warp-loop']} label="From chapter 4" />
      <Ask q={questionsS170['ch17/start/q-stride16']} label="From chapter 6" />
      <Ask q={questionsS170['ch17/start/q-lost-update']} label="From chapter 9" />
      <Ask q={questionsS170['ch17/start/q-order']} label="From appendix A" />
      <Ask q={questionsS170['ch17/start/q-scan']} label="From chapter 11" />

      <h2>The chapter in this course</h2>
      <p>
        Two teaching pages, an optional lab and a summary. Page <a href={href('ch17', '17.2')}>17.2</a> builds COO, CSR,
        ELL and CSC from the 4 × 4 example and shows that each format decides its kernel: which one needs atomics, which loads
        coalesce, which threads diverge. Page <a href={href('ch17', '17.5')}>17.5</a> buys regularity: padding, spilling
        long rows to COO, sorting rows by length, when that preprocessing pays, and where the same moves appear in LLM
        inference. The <a href={href('ch17', 'labs')}>lab</a> runs CSR and ELL SpMV on skewed rows on your own GPU. The{' '}
        <a href={href('ch17', 'wrap')}>summary</a> has a mixed check, the book's five exercises and the slips in the PDF.
      </p>
      <Source section="17.2–17.9" pages="404–422" />
      <p>
        Two chapters build on this one. Chapter 18 stores graphs in CSR, CSC and COO, and its traversal kernels look like
        these SpMV kernels. Chapter 20 names mixture of experts among the ways to scale a model.
      </p>
      <Source section="18.1, 18.4, 20.7" pages="427, 438, 510" />
      <p>
        The lab runs on every backend, because one thread per row needs no atomics and nothing vendor-specific.
      </p>
      <Today checked="2026-11-20 (ch17-facts B1, C1)">
        <p>
          HIP compiles the per-row kernels unchanged; in Metal, <code>[[thread_position_in_grid]]</code> gives the row
          directly.
        </p>
      </Today>
    </>
  );
}
