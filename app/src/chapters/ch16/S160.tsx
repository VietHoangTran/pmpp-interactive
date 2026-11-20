import { Ask, type Question } from '../../components/Ask';
import { Beyond, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { inputsOf, presetFronts, stepAt } from '../../sim/wavefront';

// The start page of chapter 16: a question before any theory, what a wavefront is and why
// the course keeps it (recurrences along a sequence in real inference code), what the
// course keeps and leaves out, and a check on the ideas from chapters 2, 4, 5, 6 and 11
// that the chapter builds on. Every number about wavefronts comes from src/sim/wavefront.ts.

// Edit distance of two 1,000-token sequences: the 1,000 x 1,000 interior of its table.
const HOOK = presetFronts('smith-waterman', 1000, 1000);
// The small table drawn on this page.
const FIG = presetFronts('smith-waterman', 6, 6);
const FIG_FRONT = FIG.sizes.indexOf(FIG.widest); // the widest step
const FIG_CELL: [number, number] = [3, 4];

export const questionsS160: Record<string, Question> = {
  'ch16/start/q-hook': {
    kind: 'choice',
    id: 'ch16/start/q-hook',
    prompt: (
      <>
        The edit distance between two sequences of 1,000 tokens fills a table of 1,001 × 1,001 cells. Row 0 and column 0
        are known before you start. Every other cell needs three cells: the one above it, the one to its left and the
        one above-left. Which is true?
      </>
    ),
    options: [
      {
        label: 'It is inherently sequential: one cell at a time.',
        why: 'Many cells need nothing from each other. Look at two cells whose row plus column is the same.',
      },
      {
        label: 'Each row can be computed in parallel once the row above is done.',
        why: 'Each cell also needs its left neighbour, which is in the same row. A row is a chain.',
      },
      {
        label: `Cells on one anti-diagonal need nothing from each other, so the table takes ${HOOK.steps.toLocaleString('en-GB')} steps of up to ${HOOK.widest.toLocaleString('en-GB')} cells.`,
        correct: true,
      },
      {
        label: 'Every cell is independent once the two sequences are known.',
        why: 'A cell needs the results of three other cells, not only the two tokens it compares.',
      },
    ],
    explain: (
      <p>
        Number the cells by row r and column q. The three inputs of a cell lie on the anti-diagonals r + q − 1 (above and
        left) and r + q − 2 (above-left), never on its own, so all cells with the same r + q can run together once the
        two anti-diagonals before them are done. The computed cells have r and q from 1 to 1,000, so r + q runs from 2 to
        2,000: {HOOK.steps.toLocaleString('en-GB')} steps, the widest {HOOK.widest.toLocaleString('en-GB')} cells. On
        average a step has only about {Math.round(HOOK.mean)} cells, half the widest, because the anti-diagonals grow
        and then shrink. That waste is what page <a href={href('ch16', '16.6')}>16.6</a> removes. The set of cells that
        can run together is a <b>wavefront</b>.
      </p>
    ),
    source:
      'ch. 16 §16.3 and §16.5, pp. 376–377, 383–384 (edit distance is our example; its table has the three-neighbour shape of the book\'s Smith-Waterman)',
  },
  'ch16/start/q-stream-order': {
    kind: 'choice',
    id: 'ch16/start/q-stream-order',
    prompt: (
      <>
        The book's Floyd-Warshall host loop launches a kernel once for every step k and calls{' '}
        <code>cudaDeviceSynchronize()</code> after each launch. Is that call needed for launch k + 1 to see what launch
        k wrote?
      </>
    ),
    options: [
      {
        label: 'Yes: without it, launch k + 1 could start before launch k has finished.',
        why: 'Not within one stream. Operations in a stream run in the order they were issued, and none can overtake another.',
      },
      {
        label: 'No: launches in one stream run in order. The call only makes the host wait.',
        correct: true,
      },
      {
        label: 'No, because a kernel launch does not return until the kernel has finished.',
        why: 'A launch returns before the kernel runs. That is why the host must wait before it reads results, but the next launch does not need the host to wait.',
      },
      {
        label: 'Only if the kernel uses atomics.',
        why: 'Atomics order accesses within a kernel. The order between launches comes from the stream.',
      },
    ],
    explain: (
      <p>
        A launch returns at once and the work goes into a stream; the operations of one stream run one after another, in
        the order they were issued. HIP documents the same rule. On Metal, dispatches in one compute encoder created
        the plain way run serially in dispatch order, and buffers created from the device are hazard-tracked, so each
        dispatch sees the previous one's writes. So the relaunch is the barrier between steps, and the extra call only
        holds the host back after every step. The corrected listing on page <a href={href('ch16', '16.3')}>16.3</a>{' '}
        leaves it out.
      </p>
    ),
    source: 'ch. 2 §2.2 and §2.6, pp. 24–25, 37–39; ch02-facts A5, B3, C9 (cited in ch16-facts, "Already settled")',
  },
  'ch16/start/q-exit-barrier': {
    kind: 'choice',
    id: 'ch16/start/q-exit-barrier',
    prompt: (
      <>
        A kernel begins with <code>if (col &gt;= n) return;</code>, and the threads that remain call{' '}
        <code>__syncthreads()</code> a few lines later. In the last block of each row some threads return. Where is this
        documented to work?
      </>
    ),
    options: [
      {
        label:
          'On CUDA, where the barrier waits only for threads that have not exited. HIP does not say, and Metal says so for Apple silicon only, so portable code keeps every thread to the barrier.',
        correct: true,
      },
      {
        label: 'On every backend: a thread that has returned cannot be waited for.',
        why: 'That is CUDA\'s documented rule. AMD\'s HIP documentation does not state it, and Apple states it for Apple silicon only.',
      },
      {
        label: 'Nowhere: a barrier after an early return always deadlocks.',
        why: 'On CUDA it is documented to work: the barrier waits for "all non-exited threads".',
      },
      {
        label: 'Only if the threads that return make up whole warps.',
        why: 'CUDA\'s rule counts exited threads one by one ("all non-exited threads"); it says nothing about whole warps.',
      },
    ],
    explain: (
      <p>
        The book's Floyd-Warshall kernel (Fig. 16.4) does exactly this, and the book does not comment on it. The
        corrected kernel on page 16.3 never returns before its barrier: an out-of-range thread keeps a flag and skips the
        work, which costs nothing and holds on every backend.
      </p>
    ),
    source: 'ch. 4 §4.3, pp. 70–72; ch04-facts A6, B3, C3',
  },
  'ch16/start/q-raw-only': {
    kind: 'choice',
    id: 'ch16/start/q-raw-only',
    prompt: (
      <>
        A block runs a loop. In each step every thread reads values that its neighbours wrote in the previous step, and
        writes its own result to a shared-memory location that no thread has used before. How many{' '}
        <code>__syncthreads()</code> calls does each step need?
      </>
    ),
    options: [
      {
        label: 'None: each thread writes only its own location.',
        why: 'Its neighbours read that location in the next step. Without a barrier they may read it before it is written.',
      },
      { label: 'One, between one step\'s writes and the next step\'s reads.', correct: true },
      {
        label: 'Two, one after the writes and one after the reads, as in the tiled matrix multiplication.',
        why: 'The second barrier there exists because the tiles are reused: a write must not overwrite what is still being read. Here no location is written twice.',
      },
      {
        label: 'None, if neighbouring threads are in the same warp.',
        why: 'Threads at the edge of a warp have neighbours in another warp, and only a barrier orders two warps of a block.',
      },
    ],
    explain: (
      <p>
        The barrier guards a true dependence: readers need what writers produce (read after write). The false one,
        write after read, appears only when a location is reused, and here none is. That is the barrier inside a tiled
        wavefront: each step writes new cells of the tile, and the next step reads them (Figs. 16.9 and 16.16 put one{' '}
        <code>__syncthreads()</code> at the end of each step inside a tile).
      </p>
    ),
    source: 'ch. 5 §5.4, pp. 110–111; ch. 16 §16.6, pp. 388–389',
  },
  'ch16/start/q-stride16': {
    kind: 'numeric',
    id: 'ch16/start/q-stride16',
    prompt: (
      <>
        The 32 threads of a warp read a shared <code>int</code> array at index <code>16*t + c</code>, where{' '}
        <code>t</code> is the lane and <code>c</code> is the same for every lane. Shared memory has 32 banks of 4 bytes.
        The access is an n-way bank conflict. What is n?
      </>
    ),
    answer: 16,
    explain: (
      <p>
        The bank is (16t + c) mod 32. Even lanes all land in bank c, odd lanes in bank c + 16: 16 lanes on each of two
        banks, so each bank serves 16 different words one after another. The book gives the same case: a tile 16 columns
        wide, read down a column. Page 16.6 meets it again: the sheared tile is stored as a square and read down a
        column, with a stride equal to the tile's width, and the book pads the rows to fix it.
      </p>
    ),
    source: 'ch. 6 §6.4, pp. 139–141 (the 16-column case, p. 141); infra-facts A10',
  },
  'ch16/start/q-counter': {
    kind: 'choice',
    id: 'ch16/start/q-counter',
    prompt: (
      <>
        Chapter 11's single-pass scan takes each block's index from an atomic counter instead of using{' '}
        <code>blockIdx.x</code>. Why?
      </>
    ),
    options: [
      { label: 'So that a block only ever waits for a block that has already started.', correct: true },
      {
        label: 'To make the flag update atomic.',
        why: 'The flag is its own atomic. The counter decides which block waits for which.',
      },
      {
        label: 'Because blockIdx.x is not unique when many blocks run at once.',
        why: 'It is unique. What nothing promises is the order in which blocks start.',
      },
      {
        label: 'To balance the work between SMs.',
        why: 'Every block does the same work. The point is who waits for whom.',
      },
    ],
    explain: (
      <p>
        Nothing promises that block <code>blockIdx.x − 1</code> has started, and a block that spins keeps its slot. If
        later blocks fill every slot and wait for one that has not started, nothing ever finishes. CUDA documents that a
        block which has started keeps making progress, so waiting only on blocks that claimed their index earlier is
        safe. Page 16.3 applies the same argument to tiles of a wavefront.
      </p>
    ),
    source: 'ch. 11 §11.9, pp. 276–282 (the counter, p. 280); ch11-facts A6, C5; ch04-facts A5',
  },
};

// ---------------------------------------------------------------- a small table, numbered by step
const CELL = 40;
const X0 = 34;
const Y0 = 30;

function SmallTable() {
  const ins = inputsOf(FIG, ...FIG_CELL);
  const cx = (c: number) => X0 + c * CELL + CELL / 2;
  const cy = (r: number) => Y0 + r * CELL + CELL / 2;
  return (
    <figure style={{ margin: '0.4rem 0 1.4rem' }}>
      <svg
        viewBox={`0 0 ${X0 + FIG.cols * CELL + 196} ${Y0 + FIG.rows * CELL + 12}`}
        role="img"
        aria-label={`A ${FIG.rows} by ${FIG.cols} table, each cell numbered by the step at which it can run, from 0 to ${FIG.steps - 1}. Step ${FIG_FRONT} is outlined: ${FIG.widest} cells.`}
        style={{ width: '100%', maxWidth: 500, display: 'block', fontFamily: 'var(--sans)' }}
      >
        <defs>
          <marker id="s160-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--ink)" />
          </marker>
        </defs>
        <text x={X0} y={16} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
          row 0 and column 0 of the full table sit above and to the left
        </text>
        {FIG.step.map((s, i) => {
          const r = Math.floor(i / FIG.cols);
          const c = i % FIG.cols;
          const front = s === FIG_FRONT;
          return (
            <g key={i} data-cell={`${r},${c}`} data-step={s}>
              <rect
                x={X0 + c * CELL + 1}
                y={Y0 + r * CELL + 1}
                width={CELL - 2}
                height={CELL - 2}
                rx="2"
                fill="var(--ink)"
                fillOpacity={0.04 + (0.3 * s) / (FIG.steps - 1)}
                stroke={front ? 'var(--ink)' : 'var(--rule)'}
                strokeWidth={front ? 2.4 : 1}
              />
              <text x={cx(c)} y={cy(r) + 4} fontSize="13" textAnchor="middle" style={{ fill: 'var(--ink)' }}>
                {s}
              </text>
            </g>
          );
        })}
        {ins.map(([r, c]) => (
          <line
            key={`${r},${c}`}
            data-input={`${r},${c}`}
            x1={cx(c) + (FIG_CELL[1] - c) * 9}
            y1={cy(r) + (FIG_CELL[0] - r) * 9}
            x2={cx(c) + (FIG_CELL[1] - c) * (CELL - 13)}
            y2={cy(r) + (FIG_CELL[0] - r) * (CELL - 13)}
            stroke="var(--ink)"
            strokeWidth="1.6"
            markerEnd="url(#s160-arrow)"
          />
        ))}
        <text x={X0 + FIG.cols * CELL + 12} y={Y0 + 18} fontSize="12" style={{ fill: 'var(--ink)' }}>
          <tspan x={X0 + FIG.cols * CELL + 12} dy="0">outlined: step {FIG_FRONT},</tspan>
          <tspan x={X0 + FIG.cols * CELL + 12} dy="16">{FIG.widest} cells that need</tspan>
          <tspan x={X0 + FIG.cols * CELL + 12} dy="16">nothing from each other</tspan>
          <tspan x={X0 + FIG.cols * CELL + 12} dy="28">arrows: the three inputs</tspan>
          <tspan x={X0 + FIG.cols * CELL + 12} dy="16">of the cell at step {stepAt(FIG, ...FIG_CELL)}</tspan>
          <tspan x={X0 + FIG.cols * CELL + 12} dy="28">{FIG.steps} steps in all</tspan>
        </text>
      </svg>
      <figcaption className="muted" style={{ fontFamily: 'var(--sans)', fontSize: '0.8rem' }}>
        The top-left {FIG.rows} × {FIG.cols} cells of such a table, each numbered by the step at which it can run: one
        more than the latest of its three inputs. Darker means later. Our drawing, computed by the model behind the
        explorer on page 16.3.
      </figcaption>
    </figure>
  );
}

export function S160() {
  return (
    <>
      <p>
        Most kernels so far had all of their work ready at the start: any thread could run first, in any order. This
        chapter is about computations where that is false. Each cell of a table needs cells computed before it, so at any moment only some
        of the work is ready. The cells that are ready together form a <b>wavefront</b>, and the kernel runs the table one
        wavefront after another.
      </p>
      <Source section="16 (opening), 16.1" pages="373–374" />

      <h2>A question before any theory</h2>
      <p>
        Try this before reading on. It is the chapter in one example, and it needs nothing but the recurrence.
      </p>
      <Ask q={questionsS160['ch16/start/q-hook']} label="Before any theory" />

      <h2>What the chapter is about</h2>
      <p>
        The book's vehicle is dynamic programming: a problem split into smaller sub-problems that overlap, so that each
        answer is reused many times and is worth keeping in a table. Filled from the smallest sub-problem up, the table
        becomes a set of cells with dependencies between them, and the cells whose inputs are all ready can be computed
        at once. The book lists shortest paths, chain matrix multiplication, dynamic time warping, the Viterbi algorithm
        of hidden Markov models and sequence alignment as problems of this kind.
      </p>
      <Source section="16.1" pages="373–374" />
      <p>
        Here is the table from the question, cut down to its top-left corner. Each cell carries the step at which it can
        run.
      </p>
      <SmallTable />
      <p>
        Fibonacci is the degenerate case. Each number needs the two before it, so every wavefront is one cell and nothing
        can run in parallel. The tables worth a GPU have wavefronts many cells wide, and the chapter is about keeping a
        GPU busy on them while still waiting for exactly what each cell needs.
      </p>
      <Source section="16.1, 16.2" pages="374, 376" figs="Eq. 16.1" />

      <h2>Why it is in this course</h2>
      <p>
        The chapter's two case studies, Floyd-Warshall's shortest paths and Smith-Waterman's sequence alignment, are not
        inference operations, and you will not write either one for a language model. The reason to read it is a kind of
        loop you will meet in inference code: a recurrence along the sequence, where step t needs step t − 1. The
        recurrent state of an RWKV or Mamba layer, the gated delta rule of linear-attention models and the CTC loss are
        all of this kind. Each is a wavefront computation, and each project picks a different way to wait between steps.
        This chapter gives you the words to read those choices.
      </p>
      <Beyond>
        <p>
          The book does not mention recurrent layers, linear attention or CTC. The link, and the code read for it, are
          ours (ch16-facts, group D).
        </p>
      </Beyond>
      <Today checked="2026-11-20 in the projects' source (ch16-facts D1 to D3)">
        <p>
          PyTorch, CTC loss: <code>LossCTC.cu</code> on CUDA and <code>LossOps.metal</code> on MPS, a barrier per time
          step.
        </p>
        <p>
          llama.cpp: <code>wkv.cu</code> and <code>wkv.metal</code> (RWKV), <code>ssm-scan.cu</code> (Mamba; a warp per
          chain in one path, chunks joined by launches in another).
        </p>
        <p>
          mlx-lm: <code>kernels/gated_delta.py</code>, a SIMD-group per chain with no threadgroup barrier in the loop.
        </p>
      </Today>

      <h2>What the course keeps</h2>
      <p>
        Two ideas. First, synchronise exactly the dependencies and no more, at the cheapest level that covers them: a
        relaunch, a block barrier, a flag between two blocks, or nothing at all because one warp owns the whole chain.
        Second, reshape the iteration space so that every step has full-width work: first cut the table into tiles, then
        shear the tiles so that their wavefronts all have the same length.
      </p>
      <p>
        Page <a href={href('ch16', '16.3')}>16.3</a> covers the first: how to number cells by the step at which they can
        run (§16.2 and §16.3), Floyd-Warshall's one launch per step and why it needs no second buffer (§16.4), and who
        waits for whom between tiles (§16.8). Page <a href={href('ch16', '16.6')}>16.6</a> covers the second on
        Smith-Waterman's table: square tiles in shared memory, then sheared ones. The{' '}
        <a href={href('ch16', 'labs')}>lab</a> builds both kinds of tile on your own GPU.
      </p>
      <p>
        What is left out: the genomics background and the scoring scheme of Smith-Waterman (§16.5, pp. 382–384), the
        traceback that recovers an alignment (p. 384), the stencil sidebar (p. 397), and the further optimisations of
        §16.8 as code: the cooperative-groups version, the flag-based kernel and the DPX instruction (pp. 397–399) appear
        as ideas and schedules, not as kernels. The <a href={href('ch16', 'wrap')}>summary</a> maps the book's six
        exercises to where each is worked and lists the slips in the PDF.
      </p>

      <h2>Before you start</h2>
      <p>
        Five questions on what this chapter takes from chapters 2, 4, 5, 6 and 11, in the order the chapter uses them.
        Each one comes back in <a href={href('review')}>review</a>.
      </p>
      <Ask q={questionsS160['ch16/start/q-stream-order']} label="From chapter 2" />
      <Ask q={questionsS160['ch16/start/q-exit-barrier']} label="From chapter 4" />
      <Ask q={questionsS160['ch16/start/q-raw-only']} label="From chapter 5" />
      <Ask q={questionsS160['ch16/start/q-stride16']} label="From chapter 6" />
      <Ask q={questionsS160['ch16/start/q-counter']} label="From chapter 11" />
    </>
  );
}
