import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { EX1, outNeighbours, runBFS, toCSR, type Kernel, type LevelRun } from '../../sim/bfs';
import { gridSummary, strideAssignment, treeSummary } from '../../sim/gridbfs';

// Chapter 18's summary page: what to carry forward, a mixed check on situations the pages did
// not show, where the book's three exercises are worked (with Exercise 1's answers computed
// from src/sim/bfs.ts), the PDF's apparent slips, what the condensed chapter left out, and the
// project status. No kernel is shown here, so there is no snippet file; Wrap.test.ts pins
// every number below against the models.

// ---------------------------------------------------------------- Exercise 1, from the model
const EX1_CSR = toCSR(EX1);
/** Exercise 1a: each vertex's out-neighbours, which are the 1s of its row in the adjacency matrix. */
export const EX1_ROWS = Array.from({ length: EX1.n }, (_, v) => outNeighbours(EX1_CSR, v));
export { EX1_CSR };

const run = (k: Kernel) => runBFS(EX1, { variant: k, root: 0, blockSize: 4, warpWidth: 4 });
/** Exercise 1c: one row per kernel and count, one value per call. */
export interface Ex1cRow {
  kernel: Kernel;
  what: string;
  values: number[];
}
const per = (k: Kernel, f: (c: LevelRun) => number) => run(k).calls.map(f);
export const EX1C: Ex1cRow[] = [
  { kernel: 'push', what: 'threads launched (one per vertex)', values: per('push', (c) => c.items) },
  { kernel: 'push', what: 'threads that loop over their out-edges', values: per('push', (c) => c.pastCheck) },
  { kernel: 'pull', what: 'threads launched (one per vertex)', values: per('pull', (c) => c.items) },
  { kernel: 'pull', what: 'threads that loop (vertex still unvisited)', values: per('pull', (c) => c.pastCheck) },
  { kernel: 'pull', what: 'threads that label their vertex', values: per('pull', (c) => c.labelsMax) },
  { kernel: 'edge', what: 'threads launched (one per edge)', values: per('edge', (c) => c.items) },
  { kernel: 'edge', what: 'threads whose source is in the last level', values: per('edge', (c) => c.pastCheck) },
  { kernel: 'edge', what: 'threads that may label (destination unvisited)', values: per('edge', (c) => c.labelsMax) },
  { kernel: 'frontier', what: 'threads launched, all of which loop', values: per('frontier', (c) => c.items) },
];
const KERNEL_NAME: Record<Kernel, string> = { push: 'Vertex push', pull: 'Vertex pull', edge: 'Edge-centric', frontier: 'Frontier push' };

// ---------------------------------------------------------------- numbers the questions use
/** A 16-ary tree five levels deep, against the 1000 x 1000 grid of page 18.6. */
export const TREE = { b: 16, depth: 5 } as const;
export const TREE_SUM = treeSummary(TREE.b, TREE.depth);
export const GRID_SUM = gridSummary(1000);
/** The emulator's cooperative grid in lab 18.1: 8 blocks of 128 threads. */
export const EMU_GRID = { blocks: 8, threads: 128 } as const;
export const STRIDE_FRONTIER = 2500;
const STRIDE = strideAssignment(STRIDE_FRONTIER, EMU_GRID.blocks * EMU_GRID.threads);
export const STRIDE_MAX = Math.max(...STRIDE.map((s) => s.length));
export const STRIDE_THREE = STRIDE.filter((s) => s.length === STRIDE_MAX).length;

const n = (v: number) => v.toLocaleString('en-US');
const list = (xs: readonly number[]) => '[' + xs.join(', ') + ']';

export const questionsWrap: Record<string, Question> = {
  'ch18/wrap/q-tree-launches': {
    kind: 'numeric',
    id: 'ch18/wrap/q-tree-launches',
    prompt: (
      <>
        A tree in which every vertex has {TREE.b} children, {TREE.depth} levels deep below its root: {n(TREE_SUM.vertices)}{' '}
        vertices. You search it from the root with the per-level host loop of Fig. 18.15. How many kernel launches does
        the search make?
      </>
    ),
    answer: TREE_SUM.launches,
    unit: 'launches',
    explain: (
      <p>
        {TREE_SUM.levels} levels after the root, one launch each, plus the last launch, which finds the frontier empty:{' '}
        {TREE_SUM.launches}. The frontier grows {TREE.b}-fold per level, to {n(TREE_SUM.maxFrontier)} at the bottom. Page
        18.6's 1000 × 1000 grid has about as many vertices, {n(GRID_SUM.vertices)}, and takes {n(GRID_SUM.launches)}{' '}
        launches, none with more than {n(GRID_SUM.maxFrontier)} vertices of work. The number of launches follows the
        depth of the graph, not its size, so the fixed cost of a level is noise for the tree and may be most of the
        time for the grid: the book's contrast between social networks and road networks.
      </p>
    ),
    source: 'ch. 18 §18.5, p. 440; §18.7, p. 445; src/sim/gridbfs.ts; the tree is ours',
  },
  'ch18/wrap/q-exchange': {
    kind: 'choice',
    id: 'ch18/wrap/q-exchange',
    prompt: (
      <>
        A repetition-penalty kernel lists the distinct tokens of a sequence: each thread runs{' '}
        <code>old = atomicExch(&amp;seen[tok], 1)</code> and appends <code>tok</code> when <code>old</code> was 0. That
        lists each token once. Why can Fig. 18.14 not do the same, exchanging in <code>currLevel</code> and appending
        when the old value was <code>UNVISITED</code>?
      </>
    ),
    options: [
      {
        label:
          'An exchange always writes: a vertex labelled at an earlier level gets the current level instead. A flag\'s repeated 1 is harmless.',
        correct: true,
      },
      {
        label: 'An exchange is not atomic on device memory: only compare-and-swap is a true read-modify-write there.',
        why: 'atomicExch is an atomic read-modify-write like the others, on CUDA and on HIP, and cuda::atomic_ref has exchange() too.',
      },
      {
        label: 'Two threads exchanging at the same moment could both get UNVISITED back, because each reads the old value before the other one writes.',
        why: 'The exchanges on one location happen one after another, and each returns the value the one before it left. Only the first sees UNVISITED, so the frontier would hold each vertex once.',
      },
      {
        label: 'It would work, with the same frontier and the same levels; compare-and-swap is chosen only because it is faster.',
        why: 'Speed is not the problem. A back edge, such as 7 → 0 in the book\'s graph, would relabel vertex 0 with the current level.',
      },
    ],
    explain: (
      <p>
        Compare-and-swap writes only when the value is still the one you expected; an exchange writes whatever is
        there. For a flag every writer stores the same value, so the unconditional write is harmless and the old value
        alone decides who appends. For BFS the written value, the level, is only right for the first visit: a vertex
        found again through a back edge already holds a smaller level, and the exchange would replace it. Exactly once
        needs the condition and the write in one step, which is what Fig. 18.14 is.
      </p>
    ),
    source: 'ch. 18 §18.5, pp. 440–442, Fig. 18.14; infra-facts A1, A2 (atomicExch, exchange); the penalty kernel is ours',
  },
  'ch18/wrap/q-sampler-order': {
    kind: 'choice',
    id: 'ch18/wrap/q-sampler-order',
    prompt: (
      <>
        A top-p sampler collects the tokens above a threshold with Fig. 18.15's private queues, then takes a running sum
        of their probabilities in list order and picks the first entry whose running sum passes <code>u</code>, a number
        drawn from a fixed seed. Same logits, same seed: two runs pick different tokens. Why, and what fixes it?
      </>
    ),
    options: [
      {
        label:
          'The list order depends on the schedule, so the same u lands on another token. Sort the survivors first, or build the list with a stable filter.',
        correct: true,
      },
      {
        label: 'The atomic counter drops a token now and then, so the lists differ in content.',
        why: 'Each fetch_add hands out a different slot, and every survivor takes exactly one, private or public. The two lists hold the same tokens in different orders.',
      },
      {
        label: 'The overflow path puts some tokens in the public list twice.',
        why: 'A survivor takes one path: a private slot below the capacity, or the public counter. It is committed once either way.',
      },
      {
        label: 'Make both counters memory_order_seq_cst: there is then one total order over all their atomics, the same in every run, and so the same list order.',
        why: 'A memory order constrains what other threads see around each atomic. It does not decide which thread reaches the counter first, and that is what sets the order of the list.',
      },
    ],
    explain: (
      <p>
        Fig. 18.16 shows the public frontier grouped by block but not in block order, and inside a block the private
        slots go in the order the atomics land. For BFS that order does not matter: any order of a frontier gives the
        same levels. For an inverse-CDF sample it does, because the token chosen is a position in the list. Either make
        the order a function of the data (sort the survivors) or never let it depend on the schedule (chapter 12's
        stable filter: an exclusive scan of keep flags gives every survivor its position). llama.cpp's expert lists, on
        page 18.6, come out in token order because they rank with a ballot rather than an atomic.
      </p>
    ),
    source: 'ch. 18 §18.6, p. 444, Fig. 18.16; ch. 12 §12.2, §12.5, pp. 290–291, 297–298; the sampler is ours',
  },
  'ch18/wrap/q-stride-barrier': {
    kind: 'choice',
    id: 'ch18/wrap/q-stride-barrier',
    prompt: (
      <>
        A colleague changes Fig. 18.17 to commit after every frontier element: the block barrier, the allocation and the
        commit move inside the stride loop{' '}
        <code>for (i = rank; i &lt; numPrevFrontier; i += numThreads)</code>. Every test whose frontiers are a multiple
        of the grid's thread count passes. What goes wrong on a frontier whose size gives some threads of one block an
        extra trip and not the rest?
      </>
    ),
    options: [
      {
        label:
          'Threads of that block make different numbers of trips, so some reach __syncthreads() more often than others: undefined, in practice a hang or lost entries.',
        correct: true,
      },
      {
        label: 'Nothing: __syncthreads() waits only for the threads still inside the loop, and the others go on to the grid barrier.',
        why: 'It waits for every thread of the block. A thread that has left the loop never arrives at that barrier; it goes on to the grid barrier, where the others never arrive.',
      },
      {
        label: 'Only the grid barrier can deadlock here; a block barrier just lets the threads that arrive early wait a little longer.',
        why: 'A block barrier that some threads of the block never reach is undefined behaviour (ch. 4, Fig. 4.4), in practice often a hang.',
      },
      {
        label: 'The commits lose coalescing on the short trip but stay correct, since every entry is still copied once.',
        why: 'Correctness goes first: the barrier is no longer reached by the whole block. Coalescing is beside the point until that is fixed.',
      },
    ],
    explain: (
      <p>
        Thread r of the grid takes elements r, r + G, r + 2G and so on of a frontier of n. When n is a multiple of G
        every thread makes n / G trips; otherwise the lowest n mod G ranks make one more. When n mod G is not a multiple
        of the block size, the boundary falls inside a block, which then holds both kinds, and any barrier inside the
        loop is reached a different number of times by threads of that block. (With G = 1024, blocks of 128 and n =
        1152, block 0 makes 2 trips throughout and the others 1: no block is split, and this size happens to pass too.) That is exactly
        the rule of chapter 4: every thread of the block executes the same <code>__syncthreads()</code>, or none does.
        Fig. 18.17 keeps its block barriers outside the stride loop, and tests that only use convenient sizes cannot
        tell the two apart.
      </p>
    ),
    source: 'ch. 4 §4.3, p. 72, Fig. 4.4; ch. 18 §18.7, pp. 446–447, Fig. 18.17; the change is ours',
  },
  'ch18/wrap/q-stride-count': {
    kind: 'numeric',
    id: 'ch18/wrap/q-stride-count',
    prompt: (
      <>
        On the course's emulator, lab 18.1's cooperative grid is {EMU_GRID.blocks} blocks of {EMU_GRID.threads}{' '}
        threads. One level's frontier holds {n(STRIDE_FRONTIER)} vertices. How many of the grid's threads handle{' '}
        {STRIDE_MAX} vertices in that level?
      </>
    ),
    answer: STRIDE_THREE,
    unit: 'threads',
    explain: (
      <p>
        The grid has {n(EMU_GRID.blocks * EMU_GRID.threads)} threads, and thread r takes elements r, r + 1,024 and r +
        2,048. A third element exists while r + 2,048 &lt; {n(STRIDE_FRONTIER)}, so for ranks 0 to{' '}
        {STRIDE_THREE - 1}: {STRIDE_THREE} threads take {STRIDE_MAX}, the rest take {STRIDE_MAX - 1}. The resident grid
        is fixed, so a frontier larger than it costs more trips per thread, not more blocks. This is the uneven trip
        count that the previous question turns into a bug.
      </p>
    ),
    source: 'ch. 18 §18.7, pp. 446–447, Fig. 18.17 lines 21–22; lab 18.1 README (the emulator grid); src/sim/gridbfs.ts',
  },
};

function Ex1cTable() {
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto', marginBottom: '1.2rem' }}>
      <table className="data" style={{ margin: 0 }} data-table="ex1c">
        <thead>
          <tr>
            <th>Kernel</th>
            <th>Count</th>
            {EX1C[0].values.map((_, i) => (
              <th key={i} className="num">Call {i + 1}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {EX1C.map((r, i) => (
            <tr key={i} data-kernel={r.kernel}>
              <td>{i === 0 || EX1C[i - 1].kernel !== r.kernel ? KERNEL_NAME[r.kernel] : ''}</td>
              <td>{r.what}</td>
              {r.values.map((v, j) => (
                <td key={j} className="num">{v}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const backends = realBackends(reports);
  const lab = 'ch18/lab1_frontier';
  const passing = [...backends, 'emu'].filter((b) => labStatus(reports[b], lab, 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch18/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul data-list="carry">
        <li>
          Choose who does the scanning. Push loops over the last level's out-edges and writes to others; pull loops over
          the unvisited vertices' in-edges and writes only its own entry, so it needs no atomic and can stop at the first
          hit. Which is cheaper changes from level to level.
        </li>
        <li>
          Count wasted threads, not only parallelism. Checking every vertex or edge at every level is simple and
          synchronisation-free, and on a deep graph almost all of it is wasted. A frontier launches threads only where
          the work is.
        </li>
        <li>
          Build a work list with an atomic counter: claim a slot with <code>fetch_add</code>, then store. The order is
          whatever the schedule made it. Where a duplicate costs work, make "check and claim" one compare-and-swap,
          with a fresh expected value each time, and loop where only a weak one exists.
        </li>
        <li>
          Privatise a hot counter: a queue per block in shared memory, an overflow path, one global reservation per
          block and a coalesced commit. Every reset of a counter that others update atomically must be atomic too.
        </li>
        <li>
          Many short steps pay a fixed cost per launch. A grid barrier removes the launches only for a grid that is
          entirely resident, so size it from the occupancy query and launch it cooperatively, or use graphs; Metal has
          neither barrier nor query, and keeps one dispatch per step.
        </li>
        <li>
          Uneven work per thread holds up the grid. Map threads to edges instead of vertices, or bucket the work by size
          onto a thread, a warp or a block.
        </li>
      </ul>
      <Source section="18.9" pages="449" />

      <h2>A mixed check</h2>
      <p>
        These draw on the whole chapter and on situations the pages did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
      </p>
      <Beyond>
        <p>
          The tree, the penalty kernel, the sampler, the colleague's change and the stride count are ours. They rest on the
          book's pages and on the facts each answer names.
        </p>
      </Beyond>
      <Ask q={q['ch18/wrap/q-tree-launches']} label="Work it out" />
      <Ask q={q['ch18/wrap/q-exchange']} />
      <Ask q={q['ch18/wrap/q-sampler-order']} />
      <Ask q={q['ch18/wrap/q-stride-barrier']} label="Bug hunt" />
      <Ask q={q['ch18/wrap/q-stride-count']} label="Work it out" />

      <h2>The book's exercises</h2>
      <p>
        The chapter ends with three exercises (pp. 449–450). The book prints no answers, so the answers in this course are
        ours, and Exercise 1's rest on our reading of the arrowheads in its drawing.
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
              <td>1a</td>
              <td>The adjacency matrix of the 8-vertex graph</td>
              <td>
                the format lens on the <a href={href('ch18', 'start')}>start page</a> with the "Exercise 1 graph" chosen: a
                vertex's out-edges are the 1s of its row. Below, row by row
              </td>
            </tr>
            <tr>
              <td>1b</td>
              <td>The graph in CSR, neighbour lists sorted</td>
              <td>
                the same lens; the CSR reading question on the <a href={href('ch18', 'start')}>start page</a> teaches the
                method. <code>srcPtrs = {list(EX1_CSR.srcPtrs)}</code>, <code>dst = {list(EX1_CSR.dst)}</code>
              </td>
            </tr>
            <tr>
              <td>1c</td>
              <td>BFS from 0: threads launched, looping and labelling per call, for four kernels</td>
              <td>
                the two predictions and the explorer "One graph, five ways to run a level" on{' '}
                <a href={href('ch18', '18.3')}>18.3</a>, with the Exercise 1 graph loaded. The full table is below
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td>Host code for direction-optimised BFS</td>
              <td>
                worked as a decision question, not as host code: the Exercise 2 question and the explorer's
                direction-optimised mode with its switch slider, on <a href={href('ch18', '18.3')}>18.3</a>
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td>Fig. 18.17 with one grid barrier per level</td>
              <td>
                the stepper "Three grid barriers" (its Exercise 3 mode), the Exercise 3 question and the "Exercise 3
                kernel" tab of the launch listing, on <a href={href('ch18', '18.6')}>18.6</a>; the lab's "Go further"
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        Exercise 1a, row by row: {EX1_ROWS.map((r, v) => `${v} → {${r.join(', ')}}`).join('; ')}. That is 15 ones in an 8 ×
        8 matrix, the same 15 entries that make up <code>dst</code>.
      </p>
      <Source section="18.10" pages="449–450" />
      <p>
        Exercise 1c, as the course's model counts it, from root 0. The search labels {'{2, 5}'}, then {'{1, 3, 7}'}, then{' '}
        {'{4, 6}'}, and needs a fourth call to find that nothing is left. Launched threads are counted one per item,
        before rounding up to whole blocks.
      </p>
      <Aside>
        <p>
          Ours, not the book's: the edge directions are our reading of p. 450, and the model's unit tests pin these counts
          against the notes.
        </p>
      </Aside>
      <Ex1cTable />
      <p>
        Read the table along a row and the work-efficiency argument of 18.3 shows up in small. Push launches 8 threads
        every call and uses at most 3; edge-centric launches 15 and, in call 3, lets four of them write two vertices,
        the benign duplicate; the frontier version launches exactly the threads that loop. Exercise 2's missing piece,
        when to switch from push to pull, the book does not give; cuGraph's rule is on 18.3.
      </p>

      <h2>Where the PDF looks wrong</h2>
      <p>
        The copy of the book this course was built from has a few apparent slips in chapter 18. They are listed so that you
        are not thrown by them when you read the book alongside.
      </p>
      <ul data-list="errata">
        <li>
          p. 441 calls "the vertex-centric push kernel without frontiers" Fig. 18.12; it is Fig. 18.6, whose lines 09 and
          10 are the check and the label.
        </li>
        <li>
          p. 439: "the threads whose vertices are not in the current level do not do anything" means the previous level,
          as line 05 of Fig. 18.6 shows.
        </li>
        <li>
          p. 433 sets <code>UNVISITED</code> to <code>cuda::std::numeric_limits&lt;unsigned&gt;</code>, which names the
          trait and not its value; the full expression is <code>cuda::std::numeric_limits&lt;unsigned int&gt;::max()</code>.
        </li>
        <li>p. 439 prints the pull bound with an unbalanced parenthesis, "O(d · n + d · m))".</li>
        <li>Fig. 18.5's caption reads "breath-first search".</li>
        <li>p. 445: "thread blocks that are not schedule" means scheduled.</li>
        <li>
          Figs. 18.7, 18.9, 18.11 and 18.13 depict the call from level 1 to level 2 but print every vertex's final level,
          8 included, which is still unvisited at that point. The explorer on 18.3 shows levels as of the call.
        </li>
        <li>
          Fig. 18.15 line 29 and Fig. 18.17 line 35 reset the private counter with a plain store while other threads of the
          block may be doing <code>fetch_add</code> on it. That is a data race; page{' '}
          <a href={href('ch18', '18.6')}>18.6</a> and the lab store through the <code>atomic_ref</code> or clamp after the
          barrier.
        </li>
        <li>
          p. 431 says the J-shaped obstacle "connects three net terminals", but Fig. 18.5 marks only the two terminals of
          the wire being routed.
        </li>
        <li>
          p. 436 calls social networks and maze-routing graphs undirected, after p. 428 described follower graphs with
          directed edges. Not a contradiction (friend graphs and follower graphs differ), but worth reading carefully.
        </li>
      </ul>
      <p>And what the listings leave to you, or write in a way that does not carry to every backend, which a real build needs:</p>
      <ul data-list="gaps">
        <li>
          No kernel checks the public frontier's capacity, and no host code allocates it. With the compare-and-swap,{' '}
          <code>numVertices</code> entries are enough; the lab checks guard words after each buffer.
        </li>
        <li>
          No host loop for the per-level kernels: resetting the counter, launching, reading the size back and swapping
          the frontiers. The lab's <code>launchBfsPrivatised</code> is one.
        </li>
        <li>
          No value for <code>PRIVATE_FRONTIER_CAPACITY</code> or <code>numThreadsPerBlock</code>; the lab uses 256 and
          128.
        </li>
        <li>
          No <code>CSRGraph</code>, <code>CSCGraph</code> or <code>COOGraph</code> struct, and no{' '}
          <code>#include &lt;cuda/atomic&gt;</code> for the <code>cuda::atomic_ref</code> of Figs. 18.12 to 18.17.
        </li>
        <li>
          <code>grid.thread_rank()</code> and <code>grid.num_threads()</code> return 64-bit values on CUDA, which Fig.
          18.17 narrows to <code>unsigned int</code> without a word; harmless for any grid small enough to be resident.
        </li>
        <li>
          The Exercise 1 graph is an unnumbered figure with small arrowheads; the 15 edges used here are our reading.
        </li>
        <li>
          p. 446 launches with four arguments. That is valid CUDA, whose last two parameters have defaults, but does not
          compile on HIP; the course passes all six.
        </li>
        <li>
          Fig. 18.17 declares its <code>__shared__</code> variables inside the level loop. That is legal, and they live as
          long as the block either way; the lab declares them at the top, where their lifetime is plain to see.
        </li>
      </ul>
      <Aside>
        <p>
          These are our readings of one PDF (notes/ch18.md, "Unclear in the PDF"), with the vendor points checked on
          2026-11-20 (notes/today/ch18-facts.md A1, A4, A7, A9, B1). If your copy differs, trust your copy.
        </p>
      </Aside>

      <h2>What this chapter left out</h2>
      <p>
        The chapter is condensed in this course, so here is what is not covered and where to find it. The adjacency-matrix
        and degree-distribution background of §18.1 (pp. 425–429) gets a paragraph and the format lens on the start page.
        Maze routing in chip design, §18.2 (pp. 430–431), with its net terminals, blockages and tie-breaking heuristics, is
        reduced to a paragraph on the start page and the wavefront animation on 18.3. Small-world graphs (p. 436) are named only, and the book's switch
        point for direction-optimised BFS, which it does not give, is replaced by cuGraph's rule in a note on 18.3; no
        direction-optimised host program is written. The linear-algebra formulation and GraphBLAS (p. 438) get one
        paragraph.
      </p>
      <p>
        Degree bucketing, §18.8 (pp. 448–449), is prose with no kernel and no lab. Graphs larger than GPU memory and
        vertex reordering (p. 449) are named only, and RAPIDS cuGraph (p. 449) is a note. The pull, edge-centric and
        direction-optimised kernels are snippets and explorer modes on 18.3, never lab work; the lab writes the frontier
        kernels and provides push and the unprivatised frontier as baselines. Exercise 2 is worked as a decision
        question, not as host code. References 1 (Kepner and Gilbert, 2011) and 3 (Luo, Wong and Hwu, 2010) on p. 451
        were not checked.
      </p>
      <Today checked="2026-11-20 (notes/today/ch18-facts.md A11, A12, A13)">
        <p>
          References 2, 4 and 5 were checked: SuiteSparse:GraphBLAS, the 2012 paper on scalable traversal, and cuGraph.
        </p>
      </Today>
      <p>Nothing later in the course depends on what was left out.</p>

      <h2>The project so far</h2>
      <p>
        This chapter adds no milestone to the through-line project. An inference engine traverses no graph, and BFS is on
        nobody's hot path. What the chapter teaches returns as patterns: work lists built with an atomic counter, as in
        the top-p candidate list of milestone M5; private queues with a single commit, as in llama.cpp's expert routing;
        and the cost of many small launches, which decoding pays every token. Lab 18.1 is optional.{' '}
        <span data-lab181={passing.join(',') || 'none'}>
          {backends.length === 0 && !reports['emu']
            ? 'No lab results are recorded on this machine yet.'
            : passing.length
              ? `Your own version passes on ${passing.map(where).join(' and ')} on this machine.`
              : 'Your own version has not passed on this machine yet.'}
        </span>{' '}
        See the <a href={href('ch18', 'labs')}>lab</a>.
      </p>
      <p>
        Chapter 19 turns to convolutional networks, and its main move is one this chapter only named: find the view in
        which an operator is one matrix multiplication and hand it to a tuned routine. A BFS level written as a sparse
        matrix–vector product is that move for graphs; a convolution unfolded into a GEMM is that move for images. Start
        at <a href={href('ch19', 'start')}>chapter 19</a>.
      </p>
      <Source section="18.4" pages="438" />
    </>
  );
}
