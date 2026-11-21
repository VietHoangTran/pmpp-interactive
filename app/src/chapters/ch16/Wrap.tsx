import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { presetFronts, stencil, wavefronts } from '../../sim/wavefront';
import { launches, stepsPerTile } from '../../sim/wavetile';

// Chapter 16's summary page: what to carry forward, a mixed check on situations the pages
// did not show, where the book's exercises are worked, the PDF's apparent slips, what the
// condensed chapter left out, and the project status. No kernel is shown for a question
// here, so there is no snippet to tie.

// The H.264 question: square tiles over a table whose cells read W and NE. A tile then reads
// the tile on its left, the tile above and the tile above-right: the stencil W, N, NE one
// level up. src/sim/wavefront.ts numbers the tiles.
const H264_T = 4;
const H264_TILES = wavefronts(H264_T, H264_T, stencil(['W', 'N', 'NE']));
const SW_TILES = presetFronts('smith-waterman', H264_T, H264_T);

// Book exercise 2: rectangular tiles of RECT.H rows (one thread each) by RECT.W columns.
const RECT = { Lseq: 1000, H: 16, W: 64 };
const RECT_ROWS = Math.ceil(RECT.Lseq / RECT.H); // 63 tile rows
const RECT_COLS = Math.ceil(RECT.Lseq / RECT.W); // 16 tile columns
const RECT_LAUNCHES = RECT_ROWS + RECT_COLS - 1; // 78
const RECT_STEPS = RECT.H + RECT.W - 1; // 79
const SQ16 = { launches: launches({ Lseq: RECT.Lseq, W: 16, scheme: 'square' }), steps: stepsPerTile({ W: 16, scheme: 'square' }) };
const SQ64 = { launches: launches({ Lseq: RECT.Lseq, W: 64, scheme: 'square' }), steps: stepsPerTile({ W: 64, scheme: 'square' }) };
const RECT_BUSY = Math.round((RECT.H * RECT.W) / RECT_STEPS); // 13
const SQ16_BUSY = Math.round((16 * 16) / SQ16.steps); // 8

export const questionsWrap: Record<string, Question> = {
  'ch16/wrap/q-h264-tiles': {
    kind: 'numeric',
    id: 'ch16/wrap/q-h264-tiles',
    prompt: (
      <>
        A table whose cells read their left neighbour and the cell above and to the right (the H.264 pattern on page
        16.3) is cut into square tiles, one block per tile, with a launch between fronts of tiles as on page 16.6. With{' '}
        {H264_T} tiles per side, what is the fewest launches that keeps every tile after the tiles it reads?
      </>
    ),
    answer: H264_TILES.steps,
    unit: 'launches',
    explain: (
      <p>
        A tile reads from its own cells, from the tile on its left, from the tile above, and, through the last cell of its
        top row, from the tile above and to the right. That last one has the same tile_row + tile_col as the tile itself,
        so page 16.6's fronts would put the two in one launch: tile (1, 0) reads a cell of tile (0, 1). Number the tiles by
        tile_col + 2·tile_row instead. That runs from 0 to {H264_TILES.steps - 1}: {H264_TILES.steps} launches of at most{' '}
        {H264_TILES.widest} tiles, against {SW_TILES.steps} launches of up to {SW_TILES.widest} for the three-neighbour
        stencil. It is the schedule of sheared tiles, <code>tile_col = d − 2·tile_row</code>, for the same reason: a
        dependency up and to the right.
      </p>
    ),
    source: 'ch. 16 §16.3, Fig. 16.2(d), and §16.7, p. 392; tiling the H.264 stencil is ours, numbered by wavefront.ts',
  },
  'ch16/wrap/q-stencil-barriers': {
    kind: 'choice',
    id: 'ch16/wrap/q-stencil-barriers',
    prompt: (
      <>
        The book's stencil sidebar repeats <code>A[i] = 0.5 * (A[i] + A[i+1])</code> for every i, once per time step. One
        block holds A in shared memory, one thread per i, and updates it in place. How many{' '}
        <code>__syncthreads()</code> does each time step need?
      </>
    ),
    options: [
      {
        label: 'One, after the writes, so that the next time step reads the values this step wrote.',
        why: 'Thread i − 1 reads A[i] in the same step in which thread i overwrites it. Something must hold that write back until the read is done.',
      },
      { label: 'Two: one between every thread\'s reads and its write, and one after the writes.', correct: true },
      {
        label: 'None, if each thread reads A[i] and A[i + 1] into registers before it writes.',
        why: 'That orders one thread\'s own read and write. It does nothing for thread i − 1, which may not have read A[i] yet.',
      },
      {
        label: 'None inside a warp: its threads run together, so only warp edges need care.',
        why: 'Nothing in the programming model promises that every thread of a warp reads before any of them writes, and threads at a warp\'s edge read from the next warp.',
      },
    ],
    explain: (
      <p>
        Two dependences, two barriers. Step t + 1 reads what step t wrote: read after write, one barrier after the writes.
        Inside a step, thread i overwrites A[i] while thread i − 1 still needs the old value: write after read, one barrier
        between the reads and the writes. That is the pair of barriers of the tiled matrix multiplication, here in a
        wavefront along time. Two arrays, reading one and writing the other and swapping them each step (double
        buffering), remove the second dependence and leave one barrier per step. The sidebar's own point goes further:
        tiles sheared in the time–space plane need fewer global synchronisations.
      </p>
    ),
    source: 'ch. 16 sidebar, p. 397; ch. 5 §5.4, pp. 110–111; ch. 6 double buffering (§16.4 points there, p. 380)',
  },
  'ch16/wrap/q-viterbi': {
    kind: 'choice',
    id: 'ch16/wrap/q-viterbi',
    prompt: (
      <>
        Viterbi decoding, one of the book's examples of dynamic programming: at every time step each of S states takes a
        maximum over all S states of the step before. S is too large for one block, so each step is split over 4 blocks.
        Would a flag per block let a block start step t earlier than a relaunch per step does?
      </>
    ),
    options: [
      {
        label:
          'No: each block of step t needs all four blocks of step t − 1, so it waits for all of them either way. Flags save launches, not the wait.',
        correct: true,
      },
      {
        label: 'Yes: with flags a block waits only for the blocks it reads, so it starts once its own block of step t − 1 is done.',
        why: 'It does wait only for the blocks it reads, but here it reads every state, so every block of step t − 1.',
      },
      {
        label: 'Yes, as long as blocks take their work from an atomic counter in step order, so each waits only on blocks that have already started.',
        why: 'The counter makes waiting safe on CUDA. It does not shrink what a block waits for.',
      },
      {
        label: 'No, because Viterbi has no dependency between time steps once the emission scores are known.',
        why: 'It does: each state at step t reads every state at step t − 1.',
      },
    ],
    explain: (
      <p>
        Synchronise exactly the dependencies: here a step depends on the whole previous step, all to all, the shape of
        Floyd-Warshall's steps rather than Smith-Waterman's neighbours. A wait on all four flags is a barrier over the four
        blocks, so it can only remove the launch overhead, as <code>grid.sync()</code> would. What changes the picture is
        the work per launch: one block per sequence with a block barrier per step, as PyTorch's CTC loss does, and many
        sequences side by side so that each launch fills the GPU.
      </p>
    ),
    source: 'ch. 16 §16.1, p. 374 (Viterbi listed), §16.4 and §16.8, pp. 379, 397–398; the Viterbi analysis is ours',
  },
  'ch16/wrap/q-causal': {
    kind: 'choice',
    id: 'ch16/wrap/q-causal',
    prompt: (
      <>
        The prefill of a causal attention layer computes a lower-triangular matrix of scores: row i uses keys 0 to i. Is
        that a wavefront computation in this chapter's sense?
      </>
    ),
    options: [
      {
        label:
          'No: no score reads another score, so every cell can run at once. The triangle makes the work uneven, not ordered.',
        correct: true,
      },
      {
        label: 'Yes: row i needs the rows above it, so the rows are the fronts, with a barrier after each one.',
        why: 'Row i needs keys 0 to i, which the layer below produced. It needs no result of another row.',
      },
      {
        label: 'Yes, and shearing the triangle, as §16.7 shears tiles, would make every step full width.',
        why: 'Shearing fills steps that dependencies force. Here there are no forced steps, only blocks with more cells than others.',
      },
      {
        label: 'Only during decoding, when the triangle grows by one row per token and each new row reads the scores of the row before it.',
        why: 'Decoding is sequential, but the dependency runs through the whole model, from one token\'s output to the next token\'s input. Inside one layer\'s score matrix no cell reads another.',
      },
    ],
    explain: (
      <p>
        A wavefront needs cells that read other cells' results. A score is a dot product of a query and a key; the softmax
        and the weighted sum then read only scores of the same row. So every row is independent, and the triangle only
        makes the work uneven: blocks over the top rows finish early. A recurrence along the sequence, such as the state
        of a Mamba or RWKV layer or PyTorch's CTC table, is a wavefront, with one front per token.
      </p>
    ),
    source: 'ours: the chapter\'s definition of a wavefront (§16.2–16.3, pp. 376–377) applied to attention; ch16-facts D1, D2',
  },
  'ch16/wrap/q-ex2': {
    kind: 'numeric',
    id: 'ch16/wrap/q-ex2',
    prompt: (
      <>
        The book's exercise 2 asks for rectangular tiles. Keep one thread per tile row, so a block of {RECT.H} threads
        fills a tile of {RECT.H} rows by {RECT.W} columns, and keep a launch per front of tiles. With L_seq ={' '}
        {RECT.Lseq.toLocaleString('en-US')}, how many launches does the table take?
      </>
    ),
    answer: RECT_LAUNCHES,
    unit: 'launches',
    explain: (
      <p>
        ⌈{RECT.Lseq.toLocaleString('en-US')} / {RECT.H}⌉ = {RECT_ROWS} tile rows and ⌈{RECT.Lseq.toLocaleString('en-US')} /{' '}
        {RECT.W}⌉ = {RECT_COLS} tile columns; the fronts are still tile_row + tile_col, from 0 to {RECT_ROWS - 1} +{' '}
        {RECT_COLS - 1}: {RECT_LAUNCHES} launches of at most {Math.min(RECT_ROWS, RECT_COLS)} tiles. Inside a tile the steps
        run along its anti-diagonals, {RECT.H} + {RECT.W} − 1 = {RECT_STEPS} of them, and the {RECT.H * RECT.W} cells keep
        about {RECT_BUSY} of the {RECT.H} threads busy on average, against about {SQ16_BUSY} in a 16 × 16 square. Square tiles of 16 need {SQ16.launches} launches; of 64, {SQ64.launches} launches but blocks of 64 threads
        and {SQ64.steps} steps per tile. What else changes: the tile index becomes <code>swTile[r_tile * {RECT.W} + q_tile]</code>,
        the bounds test <code>q_tile &lt; {RECT.W}</code>, there are two tile counts instead of one, and in the store{' '}
        {RECT.H} threads cover {RECT.W} columns, {RECT.W / RECT.H} per thread per row.
      </p>
    ),
    source: 'ch. 16 exercise 2, p. 399; the analysis is ours (notes/ch16.md, Exercises)',
  },
};

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = realBackends(reports);
  const lab = 'ch16/lab1_wavefront';
  const passing = [...backends, 'emu'].filter((b) => labStatus(reports[b], lab, 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch16/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul>
        <li>
          A wavefront is the set of cells whose inputs are all done. Number every cell one more than its latest input: the
          stencil alone decides how many steps a table takes and how wide each one is.
        </li>
        <li>
          Synchronise exactly the dependencies, at the cheapest level that covers them: a relaunch when the whole grid
          must wait for the whole previous front, a block barrier between steps inside a block, a flag when a tile waits
          only for its neighbours, nothing when one warp or thread owns the whole chain.
        </li>
        <li>
          Waiting across blocks is safe only on work that has already started: tiles from an atomic counter, or a
          cooperative launch, on CUDA; unverified on HIP; on Metal, one dispatch per front.
        </li>
        <li>
          An in-place update is safe when nothing a step reads changes during the step, as Floyd-Warshall's row k and
          column k do not when dist[k][k] is 0 and no weight is negative. Otherwise, two buffers.
        </li>
        <li>
          A tiled wavefront has two levels: launches between fronts of tiles, barriers between steps inside a tile. Keep
          the tile in shared memory and store it with thread = column, so each tile row leaves coalesced.
        </li>
        <li>
          Shearing a tile fills every step, W steps instead of 2W − 1, and costs 3T − 1 launches with fewer tiles each.
          Moving a layout moves bank conflicts: padding fixes the sheared tile's column walk and breaks the square tile's
          anti-diagonal one. Check the stride, and on Metal, measure.
        </li>
        <li>
          In inference code the wavefronts are recurrences along the sequence (RWKV and Mamba state, the gated delta
          rule), and PyTorch's CTC loss, a training objective, is one too. They run with a block barrier per step, a
          SIMD-group or warp per chain, or chunks joined by launches.
        </li>
      </ul>
      <Source section="16.9" pages="399" />

      <h2>A mixed check</h2>
      <p>
        These draw on the whole chapter and on situations the pages did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
      </p>
      <Beyond>
        <p>
          Tiling the H.264 stencil, the Viterbi split and the attention question are ours. The stencil loop is the book's
          sidebar; exercise 2 is the book's, its answer ours.
        </p>
      </Beyond>
      <Ask q={q['ch16/wrap/q-h264-tiles']} label="Work it out" />
      <Ask q={q['ch16/wrap/q-stencil-barriers']} />
      <Ask q={q['ch16/wrap/q-viterbi']} />
      <Ask q={q['ch16/wrap/q-causal']} label="In an LLM" />
      <Ask q={q['ch16/wrap/q-ex2']} label="Book exercise 2" />

      <h2>The book's exercises</h2>
      <p>
        The chapter ends with six exercises (pp. 399–400), and sets two more in the text. All of them ask for code; the
        book prints no answers, so the answers here are ours. Lab 16.1 builds the two set in the text; the six at the end are
        analysed, not built, and the table says where.
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
              <td>Floyd-Warshall with square blocks, row k and column k in shared memory</td>
              <td>
                the "Book exercise 1" question in <a href={href('ch16', '16.3')}>16.3</a>: reads per cell and what the count
                leaves out; not built
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td>Rectangular tiles</td>
              <td>the "Book exercise 2" question above; not built</td>
            </tr>
            <tr>
              <td>3</td>
              <td>Cooperative groups instead of a relaunch</td>
              <td>
                "Who waits for whom" in <a href={href('ch16', '16.3')}>16.3</a>: the grid barrier, what it saves and does
                not, and its form on each backend (none on Metal); the API is the book's §18.7, worked on page{' '}
                <a href={href('ch18', '18.6')}>18.6, "Private queues and one launch for every level"</a>; not built
              </td>
            </tr>
            <tr>
              <td>4</td>
              <td>Flags between square tiles</td>
              <td>
                the animation "Waiting for everyone, or only for your neighbours", the "Who waits for whom" explorer and the persistent-row question in{' '}
                <a href={href('ch16', '16.3')}>16.3</a>; not built, because only CUDA documents a safe form
              </td>
            </tr>
            <tr>
              <td>5</td>
              <td>Flags between sheared tiles</td>
              <td>
                the dependency is in <a href={href('ch16', '16.6')}>16.6</a> (a sheared tile reads the tile on its left
                and the tile above and to the right); the H.264 question above has the same one; not built
              </td>
            </tr>
            <tr>
              <td>6</td>
              <td>A DPX instruction for max4</td>
              <td>
                the DPX question in <a href={href('ch16', '16.6')}>16.6</a>; the lab's README says how to try it on an
                NVIDIA GPU
              </td>
            </tr>
            <tr>
              <td>p. 384</td>
              <td>One launch per anti-diagonal, one thread per cell</td>
              <td>
                provided in <a href={href('ch16', 'labs')}>lab 16.1</a> as the baseline, <code>labs/ch16/shared/</code>
              </td>
            </tr>
            <tr>
              <td>p. 386</td>
              <td>Launch exactly the blocks a front needs</td>
              <td>
                the exact-blocks question in <a href={href('ch16', '16.6')}>16.6</a>, and both launchers of{' '}
                <a href={href('ch16', 'labs')}>lab 16.1</a>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Where the PDF looks wrong</h2>
      <p>
        The copy of the book this course was built from has a number of apparent slips in chapter 16. They are listed so
        that you are not thrown by them when you read the book alongside. First, wording and numbering:
      </p>
      <ul data-list="errata">
        <li>
          Fig. 16.5 labels the edge from b to f 6 in the graph and 9 in the table. The course uses the table as printed (
          <a href={href('ch16', '16.3')}>16.3</a>).
        </li>
        <li>
          Eq. 16.2 subtracts the deletion penalty from the cell above and the insertion penalty from the cell on the left;
          Fig. 16.6, the text on p. 383 and the kernel attach the names the other way round. With equal penalties nothing
          changes.
        </li>
        <li>
          p. 382 indexes H by a prefix of A for i and of B for j; p. 383 and Fig. 16.6 use i for B and j for A, and p. 383's
          "the B_j was incorrectly inserted" means B_i.
        </li>
        <li>
          Fig. 16.14(a) labels the square tiles' fronts d = 0, 1, 2, 4, 5; the text numbers them 0 to 4, so the last two are
          3 and 4.
        </li>
        <li>
          p. 391 works the shear for "cell (1, 0) in tile (0, 1)"; the arithmetic is for cell (2, 0). Later on the page,
          "only tile (1, 1)" at d = 1 means tile (0, 1), as the kernel's formula and the next sentences give.
        </li>
        <li>p. 398 says max4 is "in the kernel of Fig. 16.10"; it is called in Figs. 16.9 and 16.16 and defined in Fig. 16.11.</li>
        <li>p. 377 says computing F_i needs F_i and F_i−1 kept; Eq. 16.1 needs F_i−1 and F_i−2.</li>
        <li>
          p. 396's "up to 1.5× more wavefronts" means up to 1.5 times as many: (3T − 1) / (2T − 1) approaches 1.5.
        </li>
      </ul>
      <p>And in the listings, where it affects code:</p>
      <ul data-list="code">
        <li>
          Fig. 16.4 compares an <code>int</code> with <code>INFINITY</code>, a <code>float</code> that no <code>int</code>{' '}
          equals, so the early exits never fire, and with <code>INT_MAX</code> as "no edge" the sum overflows. The course
          uses <code>INT_MAX / 2</code> (<a href={href('ch16', '16.3')}>16.3</a>).
        </li>
        <li>
          Fig. 16.4 returns out-of-range threads before its <code>__syncthreads()</code>. CUDA documents that exited threads
          do not hold a barrier; HIP does not say, and Metal says so for Apple silicon only. The course keeps every thread
          to the barrier.
        </li>
        <li>
          Figs. 16.9 and 16.16 compute <code>tile_col</code> and <code>q_tile</code> from unsigned values and store a
          possibly negative result in an <code>int</code>, a conversion C++17 leaves to the compiler. The lab computes in{' '}
          <code>int</code>.
        </li>
        <li>
          Fig. 16.18's sheared store tests only <code>q &lt; L</code>, and works because <code>L</code> is unsigned, so a
          negative q wraps round. The lab writes <code>q &gt;= 1 &amp;&amp; q &lt; L</code>.
        </li>
        <li>
          The <code>pad()</code> macro of p. 395 adds one gap per bank-count ints, not one per tile row: the same only when
          W equals the number of banks, and a 2-way conflict at W = 64 on 32 banks. The printed listings never apply it. The
          lab pads one gap per row through a stride (<a href={href('ch16', '16.6')}>16.6</a>).
        </li>
        <li>
          The listings never define <code>INFINITY</code>, <code>T</code>, <code>MATCH</code>, <code>MISMATCH</code>,{' '}
          <code>DELETION</code>, <code>INSERTION</code>, or the host snippets' <code>threads</code>, <code>L_seq</code>,{' '}
          <code>sw</code>, <code>rea</code> and <code>ref</code>. The penalties are added, so they must be negative, which
          the book implies but does not state.
        </li>
      </ul>
      <Aside>
        <p>
          These are our readings of one PDF (notes/ch16.md, "Unclear in the PDF"), with the code points checked on
          2026-11-20 (ch16-facts A3, E1; ch04-facts A6, B3, C3). Fig. 16.2(b) and the sidebar's figure are also hard to
          read at print size; the explorer's financial stencil is our reading. If your copy differs, trust your copy.
        </p>
      </Aside>

      <h2>What this chapter left out</h2>
      <p>
        The chapter is condensed in this course, so here is what is not covered and where to find it. From §16.5: the
        genomics background (p. 382), what the score means, how a scoring scheme is chosen and what insertions and
        deletions are (pp. 382–383, Eq. 16.2, Fig. 16.6), of which the lab keeps only the example values as constants, and
        the traceback that recovers an alignment (p. 384). From §16.1 to §16.4: the Fibonacci listings line by line and
        their call trace (Fig. 16.1, pp. 374–376), the application domains behind the families of problems (p. 374) and
        behind Fig. 16.2's four patterns (p. 377), both named in a sentence here, and Floyd-Warshall's graph framing and
        pseudo-code beyond its dependency pattern (Fig. 16.3, pp. 377–379).
      </p>
      <p>
        From §16.7: the cache argument for sheared tiles (Fig. 16.13, pp. 389–390), stated in a paragraph on page 16.6 and
        not modelled, and the sidebar on sheared tiles for stencils in time and space (p. 397), named there and used by one
        question above. From §16.8: the cooperative-groups kernel, the flag kernels and small problems held in registers
        with shuffles (pp. 397–398), all discussed and modelled on page 16.3 but not built, and DPX as a question only.
        Exercises 1 and 2 are analysed, not built. Of the references (p. 400), only the origin of hypertiling is cited.
        Nothing later in the course depends on what was left out.
      </p>

      <h2>The project so far</h2>
      <p>
        This chapter adds no milestone to the through-line project: nothing on the main path of a decoder-only
        transformer's step, its matrix products, attention, softmax or sampling, fills a dynamic-programming table. Its
        ideas apply to recurrent and linear-attention layers, which the project's model does not have, and to reading how
        llama.cpp, PyTorch and mlx-lm run them. Lab 16.1 is optional.{' '}
        <span data-lab161={passing.join(',') || 'none'}>
          {backends.length === 0 && !reports['emu']
            ? 'No lab results are recorded on this machine yet.'
            : passing.length
              ? `Your own version passes on ${passing.map(where).join(' and ')} on this machine.`
              : 'Your own version has not passed on this machine yet.'}
        </span>{' '}
        See the <a href={href('ch16', 'labs')}>lab</a>. The next milestone, M6 attention, comes with{' '}
        <a href={href('ch20', 'start')}>chapter 20</a>.
      </p>
      <p>
        Chapter 17 turns from dependencies to irregular data. A sparse matrix–vector product has no wavefronts at all, but
        the format the matrix is stored in decides whether loads coalesce, whether threads diverge and whether blocks get
        equal work, and the trade between padding and wasted work comes back as the price of a regular layout. Start at{' '}
        <a href={href('ch17', 'start')}>chapter 17</a>.
      </p>
    </>
  );
}
