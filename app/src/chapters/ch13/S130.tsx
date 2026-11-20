import { Ask, type Question } from '../../components/Ask';
import { Beyond, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { RUNNING, coRankByMerge, mergeStable, mergeWith, outputsPerBlock, tag, type Item } from '../../sim/mergetile';

// The start page of chapter 13: what the chapter is for and what the course keeps of
// it, a question before any theory, the book's running example and its tie rule, a map,
// and a check on the ideas from chapters 2, 5 and 6 that the chapter builds on.
// Every number about the running example comes from src/sim/mergetile.ts.

const A = [...RUNNING.A];
const B = [...RUNNING.B];
const C = mergeStable(A, B);
const C_LT = mergeWith(A, B, '<');
const TIE_DIFF = C.filter((x, k) => tag(x) !== tag(C_LT[k])).length; // 5
const I3 = coRankByMerge(3, A, B); // 2
const I6 = coRankByMerge(6, A, B); // 5
const list = (xs: readonly number[]) => '[' + xs.join(', ') + ']';
const seq = (items: Item[]) => items.map(tag).join(' ');

// Fig. 13.9, line 03, with m + n = 9 and 4 threads: the share as printed and as meant.
const PER_PRINTED = outputsPerBlock(9, 4, 'printed'); // 2
const PER_MEANT = outputsPerBlock(9, 4, 'corrected'); // 3
const WRITTEN = Math.min(9, 4 * PER_PRINTED); // 8

export const questionsS130: Record<string, Question> = {
  'ch13/start/q-hook': {
    kind: 'choice',
    id: 'ch13/start/q-hook',
    prompt: (
      <>
        <code>A = {list(A)}</code> and <code>B = {list(B)}</code> are sorted. <code>C</code> is their merge, 9 elements,
        and it is stable: where two keys are equal, the element from A comes first. Three threads split C evenly, three
        outputs each, so thread 1 produces <code>C[3]</code>, <code>C[4]</code> and <code>C[5]</code>. Which input
        elements does thread 1 need?
      </>
    ),
    options: [
      {
        label: (
          <>
            <code>A[1]</code>, <code>A[2]</code> and <code>A[3]</code> (7, 8 and 9), and nothing from B.
          </>
        ),
        why: 'A[1] = 7 is C[1]. The 1 and both 7s come before 8, so thread 1 starts at C[3] = 8.',
      },
      {
        label: (
          <>
            <code>A[2]</code> and <code>A[3]</code> (8 and 9), and <code>B[1]</code> (10).
          </>
        ),
        why: 'The same three values, but the tie rule decides which 10 is C[5]: A[4] = 10 goes before B[1] = 10. So C[5] is A[4], and B[1] is C[6], the first output of thread 2.',
      },
      {
        label: (
          <>
            <code>A[2]</code>, <code>A[3]</code> and <code>A[4]</code> (8, 9 and 10), and nothing from B.
          </>
        ),
        correct: true,
      },
      {
        label: 'Two elements of A and one of B, since B holds almost half the data.',
        why: 'That holds only where the two arrays interleave evenly. One thread\'s share of each array depends on the values, and it can be zero.',
      },
    ],
    explain: (
      <p>
        Count how many of the first 3 outputs come from each array: {I3} from A, {3 - I3} from B. Of the first 6:{' '}
        {I6} from A, {6 - I6} from B. Thread 1 takes the difference, A[{I3}] up to A[{I6 - 1}] and nothing from B. Those
        pairs, ({I3}, {3 - I3}) and ({I6}, {6 - I6}), are the co-ranks of 3 and 6. No arithmetic on the thread index
        produces them: they depend on the values. Page <a href={href('ch13', '13.3')}>13.3</a> finds them with a binary
        search instead of a merge.
      </p>
    ),
    source: 'ch. 13 §13.3 and §13.5, pp. 308, 314 (the book works this three-thread split)',
  },
  'ch13/start/q-tie': {
    kind: 'choice',
    id: 'ch13/start/q-tie',
    prompt: (
      <>
        Suppose <code>merge_sequential</code> compared with <code>A[i] &lt; B[j]</code> instead of{' '}
        <code>A[i] &lt;= B[j]</code>. What happens to C for the running example?
      </>
    ),
    options: [
      {
        label: 'C is no longer sorted wherever the two arrays hold equal keys.',
        why: 'Each step still copies the smaller front element. On a tie either choice keeps C sorted.',
      },
      {
        label: 'Nothing changes: equal keys cannot be told apart, so C is the same as before.',
        why: `Their keys cannot; the elements can. Tag each element with its origin and C differs in ${TIE_DIFF} of its 9 places.`,
      },
      {
        label: "B's two 10s swap places, and A's 10 moves after both of them.",
        why: "A's 10 does move after them, but within one list the merge copies in order: j only ever moves forward. Only the order across the two lists changes.",
      },
      {
        label: "Keys in the same order, but B's 7 before A's 7 and A's 10 after B's two 10s.",
        correct: true,
      },
    ],
    explain: (
      <p>
        With <code>&lt;</code> a tie sends B's element first, and C becomes <code>{seq(C_LT)}</code> instead of{' '}
        <code>{seq(C)}</code>. It is sorted and still stable within each list, but not across them. The co-rank search
        has to encode the book's choice too: one of its two tests is strict and the other is not (<code>&gt;</code> and{' '}
        <code>&gt;=</code>, p. 311).
      </p>
    ),
    source: 'ch. 13 §13.1 and §13.2, pp. 304–305 (the < variant is ours)',
  },
  'ch13/start/q-ceil': {
    kind: 'numeric',
    id: 'ch13/start/q-ceil',
    prompt: (
      <>
        A kernel gives each thread its share of the outputs with{' '}
        <code>int per = ceil((m + n) / (blockDim.x * gridDim.x));</code> and thread <code>t</code> writes outputs{' '}
        <code>t*per</code> up to <code>(t+1)*per - 1</code>. With <code>m + n = 9</code> and 4 threads in all, how many of
        the 9 outputs get written?
      </>
    ),
    answer: WRITTEN,
    unit: 'outputs',
    explain: (
      <p>
        <code>(m + n) / (blockDim.x * gridDim.x)</code> divides integers, so 9 / 4 is already {PER_PRINTED} when{' '}
        <code>ceil</code> sees it. Four threads write {PER_PRINTED} each, <code>C[0]</code> to <code>C[{WRITTEN - 1}]</code>
        , and <code>C[8]</code> is never written. Nothing fails; one output is simply missing. Chapter 2 avoided this
        with a floating-point literal, <code>ceil(n/256.0)</code>; NVIDIA teaches the integer form{' '}
        <code>(a + b - 1) / b</code>, which gives {PER_MEANT} here. The line is the book's own, from its basic merge
        kernel (Fig. 13.9, line 03, where <code>per</code> is called <code>elementsPerThread</code>), and the same slip
        makes its tiled kernel run 3 iterations where the text says 4.
      </p>
    ),
    source: 'ch. 2 §2.6, p. 38; ch02-facts A8; ch. 13 Fig. 13.9 line 03 (notes/ch13.md, "Unclear in the PDF" item 14; ch13-facts A4)',
  },
  'ch13/start/q-stride': {
    kind: 'numeric',
    id: 'ch13/start/q-stride',
    prompt: (
      <>
        On an NVIDIA GPU, global memory is accessed in 32-byte segments and a warp has 32 threads. In one load, lane{' '}
        <code>t</code> of a warp reads the 4-byte float at index <code>2*t</code> of an array whose element 0 starts a
        segment. How many segments does that one load touch?
      </>
    ),
    answer: 8,
    unit: 'segments',
    explain: (
      <p>
        The lanes read indices 0, 2, ..., 62: bytes 0 to 251, inside a 256-byte span, which is 8 segments. Half of every
        segment goes unused, so half the bytes moved are wasted; NVIDIA's guide gives the same 50% for a stride of 2.
        Page 13.3 meets a worse case: in the book's basic merge kernel each thread walks its own stretch of A and B, so
        neighbouring lanes read addresses several elements apart, by amounts that depend on the data (§13.5, p. 314).
      </p>
    ),
    source: "ch. 6 §6.1, pp. 124–132; NVIDIA's 32-byte unit and the stride-2 figure: ch06-facts A2",
  },
  'ch13/start/q-strategy': {
    kind: 'choice',
    id: 'ch13/start/q-strategy',
    prompt: (
      <>
        Every thread of a block binary-searches the same sorted 1,024-element array, which sits in global memory. Each
        search reads about ten elements, at places that depend on the key it looks for. Which change makes the block's
        global-memory reads coalesced?
      </>
    ),
    options: [
      {
        label: 'Assign keys to threads so that neighbouring threads search for neighbouring keys.',
        why: 'Their probes may land closer together, but which elements a search reads still depends on the data. Nothing lines the lanes up on consecutive addresses.',
      },
      {
        label: 'Have the block copy the array into shared memory, coalesced, and search it there.',
        correct: true,
      },
      {
        label: 'Store the array in a different layout, as you would a transposed matrix.',
        why: 'Changing the layout helps when the access pattern is fixed, as with a transposed matrix. Here the elements a search reads depend on the keys it looks for, so a fixed rearrangement cannot promise that the lanes of a warp read neighbouring addresses.',
      },
      {
        label: 'Launch more threads, so the memory system has more requests to combine.',
        why: 'More irregular requests are still irregular. Coalescing needs the lanes of one warp to hit the same segments in the same load.',
      },
    ],
    explain: (
      <p>
        This is the third of chapter 6's strategies: move the data between global and shared memory in a coalesced
        pattern, consecutive threads on consecutive elements, then make the irregular accesses in shared memory, which
        does not need coalescing. Chapter 6's
        checklist names this very chapter as its example. The tiled merge kernel of §13.6 does it with a tile of A and a
        tile of B.
      </p>
    ),
    source: 'ch. 6 §6.1, p. 129 (strategy 3), and §6.8, p. 150, which names chapter 13',
  },
  'ch13/start/q-barriers': {
    kind: 'choice',
    id: 'ch13/start/q-barriers',
    prompt: (
      <>
        A block runs a loop. In each iteration its threads load a tile into a shared array, then every thread
        binary-searches the whole tile, then the next iteration loads the next tile into the same array. Which{' '}
        <code>__syncthreads()</code> calls does each iteration need?
      </>
    ),
    options: [
      {
        label: 'One, after the loads, so that no thread searches a half-loaded tile.',
        why: 'That one keeps threads from searching a half-loaded tile. Without a second, a fast thread starts loading the next tile over elements a slower thread is still searching.',
      },
      {
        label: 'One, after the searches, so that no thread overwrites a tile in use.',
        why: 'Then a thread may search the tile before the other threads have finished writing it.',
      },
      { label: 'Two: one after the loads and one after the searches.', correct: true },
      {
        label: 'None, if a single warp both loads and searches the tile.',
        why: "On NVIDIA GPUs of compute capability 7.0 and later the threads of a warp are not guaranteed to run in lockstep (ch04-facts A10). A warp gives you no ordering to rely on; the barrier does.",
      },
    ],
    explain: (
      <p>
        The first barrier guards a true dependence: every thread reads elements that other threads wrote (read after
        write). The second guards a false one: the next loads need nothing from the searches, they only must not
        overwrite elements still being read (write after read). The tiled merge kernel has both (Fig. 13.12 line 36 and
        Fig. 13.13 line 50), and two more in Fig. 13.11 around sharing the block's co-ranks through shared memory.
      </p>
    ),
    source: 'ch. 5 §5.4, pp. 110–111; ch. 13 §13.6, pp. 316–318',
  },
  'ch13/start/q-coarsen': {
    kind: 'numeric',
    id: 'ch13/start/q-coarsen',
    prompt: (
      <>
        Every thread of a kernel makes 40 global reads of setup before it produces anything, then 1 global read per
        output. There are 1,000,000 outputs. With 8 outputs per thread, how many global reads does the whole grid make, in
        millions?
      </>
    ),
    answer: 6,
    tol: 0.001,
    unit: 'million reads',
    explain: (
      <p>
        1,000,000 / 8 = 125,000 threads pay 40 reads each, 5 million, plus 1 million for the outputs: 6 million. At one
        output per thread it is 41 million, almost all of it setup. That is thread coarsening: when parallel work costs
        something per thread, give each thread more work. In a merge the per-thread setup is the co-rank search, and the
        book coarsens every merge kernel in the chapter for that reason (§13.8, p. 327).
      </p>
    ),
    source: 'ch. 6 §6.5, pp. 141–143; ch. 13 §13.8, p. 327',
  },
};

// ---------------------------------------------------------------- the running example, redrawn
const CELL = 40;
const STEP = 46;
const X0 = 34;
const cellX = (k: number) => X0 + k * STEP;

/** Runs of equal keys in C that hold more than one element: where the tie rule decides. */
function tieGroups(c: Item[]) {
  const groups: { from: number; to: number; items: Item[] }[] = [];
  for (let k = 0; k < c.length; ) {
    let e = k;
    while (e + 1 < c.length && c[e + 1].key === c[k].key) e++;
    if (e > k) groups.push({ from: k, to: e, items: c.slice(k, e + 1) });
    k = e + 1;
  }
  return groups;
}

function Row({ name, y, cells }: { name: string; y: number; cells: Item[] }) {
  return (
    <g data-row={name}>
      <text x={8} y={y + 25} fontSize="15" fontWeight="600" style={{ fill: 'var(--ink)' }}>
        {name}
      </text>
      {cells.map((it, k) => (
        <g key={k} data-tag={tag(it)}>
          <text x={cellX(k) + CELL / 2} y={y - 6} fontSize="12" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
            {tag(it)}
          </text>
          <rect
            x={cellX(k)}
            y={y}
            width={CELL}
            height={36}
            rx={3}
            fill="var(--global-wash)"
            stroke="var(--global)"
            strokeWidth={1.5}
            strokeDasharray={it.src === 'B' ? '4 3' : undefined}
          />
          <text x={cellX(k) + CELL / 2} y={y + 24} fontSize="15" textAnchor="middle" style={{ fill: 'var(--ink)' }}>
            {it.key}
          </text>
        </g>
      ))}
    </g>
  );
}

function RunningExample() {
  const aItems: Item[] = A.map((key, idx) => ({ key, src: 'A', idx }));
  const bItems: Item[] = B.map((key, idx) => ({ key, src: 'B', idx }));
  const ties = tieGroups(C);
  const yC = 176;
  return (
    <figure style={{ margin: '0.4rem 0 1.4rem' }}>
      <svg
        viewBox="0 0 456 252"
        role="img"
        aria-label={`A is ${list(A)}, B is ${list(B)}, and their stable merge C is ${list(C.map((x) => x.key))}, taken in the order ${seq(C)}.`}
        style={{ width: '100%', maxWidth: 456, display: 'block', fontFamily: 'var(--sans)' }}
      >
        <Row name="A" y={24} cells={aItems} />
        <Row name="B" y={96} cells={bItems} />
        <Row name="C" y={yC} cells={C} />
        {ties.map((g) => {
          const x1 = cellX(g.from) + 2;
          const x2 = cellX(g.to) + CELL - 2;
          const yb = yC + 46;
          return (
            <g key={g.from} data-tie={g.items[0].key}>
              <path d={`M${x1},${yb} v6 H${x2} v-6`} fill="none" stroke="var(--ink-2)" strokeWidth={1.2} />
              <text x={(x1 + x2) / 2} y={yb + 22} fontSize="12" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
                {g.items.map(tag).join(' before ')}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption className="muted" style={{ fontFamily: 'var(--sans)', fontSize: '0.8rem' }}>
        The book's running example (Fig. 13.1), redrawn. The tag above each element names its origin (A1 is A[1]), and a
        dashed outline marks an element from B. The brackets mark equal keys, where the tie rule decides. All three
        arrays are in the global-memory colour: the chapter's kernels keep them there (p. 313).
      </figcaption>
    </figure>
  );
}

export function S130() {
  return (
    <>
      <p>
        Every kernel so far found its inputs by arithmetic on its own index. Thread <code>i</code> of a vector add reads
        element <code>i</code>; a tiled matmul block loads the tiles its <code>blockIdx</code> names. Merging two sorted
        arrays breaks that. Which inputs feed output <code>k</code> depends on the values in both arrays, so a thread has
        to search for where its inputs start before it can do any work.
      </p>
      <Source section="13 (opening), 13.3" pages="303, 308" />
      <p>
        The chapter is condensed in this course for a plain reason: no matmul, attention or quantised kernel merges two
        sorted lists. Two of its ideas do transfer, and they are what the course keeps:
      </p>
      <ul>
        <li>
          <b>the co-rank search</b>: give every thread an equal slice of the output, then binary-search for the two
          input ranges that produce it. Each thread then has the same amount of work, whatever the data looks like;
        </li>
        <li>
          <b>staging irregular reads through shared memory</b>: when a block will read its inputs at places that depend
          on the data, it first copies the part it needs into shared memory with coalesced loads, one tile at a time,
          and makes the irregular reads there.
        </li>
      </ul>

      <p>
        On an Apple GPU this search already sits under code you may use. MLX's <code>mx.sort</code> is a merge sort, and{' '}
        <code>mx.argpartition</code> and <code>mx.topk</code> go through the same sort. Its merge passes split the work
        with a co-rank, first per block on device memory, then per thread on a tile in threadgroup memory: close to the
        book's tiled kernel. mlx-lm's top-k sampler calls <code>mx.argpartition</code> and its top-p sampler{' '}
        <code>mx.sort</code>, so sampling from an MLX model with top-k or top-p runs this chapter's search. PyTorch's MPS backend has
        a merge sort built the same way behind <code>torch.sort</code> and <code>torch.topk</code>, chosen per call
        against a radix sort.
      </p>
      <Beyond>
        <p>
          The book mentions neither MLX nor PyTorch, nor sampling. The link is ours, read from the projects' source
          code (ch13-facts D1, D2).
        </p>
      </Beyond>
      <p>
        When you read that code, look for another name: the libraries call the co-rank <i>merge path</i> (CUB, rocPRIM)
        or <code>merge_partition</code> (MLX, PyTorch). And on NVIDIA GPUs the usual top-k is not a merge at all:
        PyTorch and llama.cpp use radix select or bitonic sort there.
      </p>
      <Today checked="2026-11-20 in the projects' source (ch13-facts A1, D1, D2, D4)">
        <p>
          MLX: <code>mb_block_partition</code> and <code>mb_block_merge</code> in <code>kernels/sort.h</code>. PyTorch
          MPS: <code>merge_partition</code> in <code>kernels/SortMerge.h</code>.
        </p>
      </Today>

      <h2>A question before any theory</h2>
      <p>
        Try this before reading on. It is the whole chapter in one small example, and you need nothing more than the
        merge you already know.
      </p>
      <Ask q={questionsS130['ch13/start/q-hook']} label="Before any theory" />

      <h2>The running example</h2>
      <p>
        The book uses these two arrays for the whole chapter, and so does this course. Here they are with their merge,
        each element tagged with where it came from.
      </p>
      <RunningExample />
      <p>
        The sequential merge is the one from merge sort: walk both arrays with one index each, copy the smaller front
        element to C, and when one array runs out, copy the rest of the other. Each element is touched once, so it costs
        O(m + n). The detail that matters in this chapter is the comparison, <code>A[i] &lt;= B[j]</code>: on equal
        keys, A's element goes first. That makes the merge <b>stable</b>. Equal keys keep their order within each list
        (B's two 10s stay in order) and across the lists (A's 7 before B's 7, A's 10 before both of B's).
      </p>
      <Source section="13.1, 13.2" pages="303–306" figs="Figs. 13.1, 13.2" />
      <p>
        Why keep a tie order when equal keys look the same? Because elements carry more than their key: records sorted
        earlier by another key keep that order through a stable merge, so the earlier work is not lost. The tie rule also
        reappears in the co-rank search as a pair of lopsided tests.
      </p>
      <Ask q={questionsS130['ch13/start/q-tie']} />

      <h2>The chapter in this course</h2>
      <p>
        One teaching page, an optional lab and a summary. <a href={href('ch13', '13.3')}>Page 13.3</a> covers what the
        chapter teaches after this page: the co-rank function and its binary search (§13.3 and §13.4, pp. 306–313), the
        basic kernel built on it (§13.5), the tiled kernel that stages both inputs through shared memory (§13.6), and why
        every merge kernel in the chapter is coarsened (§13.8). The <a href={href('ch13', 'labs')}>lab</a> merges two
        sorted arrays on your own GPU with co-rank partitioning. The <a href={href('ch13', 'wrap')}>summary</a> has a
        mixed check, the book's four exercises and the slips in the PDF.
      </p>
      <p>
        What is left out: §13.7, the circular-buffer kernel (pp. 321–327), which keeps the half of each tile that the
        tiled kernel throws away. Page 13.3 describes it in one paragraph, and the summary page says why the course goes
        no further. The two background sections, §13.1 and §13.2, are this page.
      </p>

      <h2>Before you start</h2>
      <p>
        Five questions on what this chapter takes from chapters 2, 5 and 6, in the order the chapter uses them. Each one
        comes back in <a href={href('review')}>review</a>.
      </p>
      <Ask q={questionsS130['ch13/start/q-ceil']} label="From chapter 2" />
      <Ask q={questionsS130['ch13/start/q-stride']} label="From chapter 6" />
      <Ask q={questionsS130['ch13/start/q-strategy']} label="From chapter 6" />
      <Ask q={questionsS130['ch13/start/q-barriers']} label="From chapter 5" />
      <Ask q={questionsS130['ch13/start/q-coarsen']} label="From chapter 6" />
    </>
  );
}
