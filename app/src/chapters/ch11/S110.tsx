import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';

// Chapter 11's start page: what scan does in an inference stack, the chapter as a cost
// ledger of the book's formulas, a prerequisite check drawn from chapters 4 to 10, and
// the six labs. The ledger is a static table of the book's formulas with their pages.

const LEDGER: { what: string; formula: string; pages: string; sec: string; secLabel: string }[] = [
  { what: 'Barriers in one block scan, Fig. 11.3', formula: '2 · log₂B', pages: 'p. 257', sec: '11.2', secLabel: '11.2' },
  { what: 'Barriers in one block scan, Fig. 11.5 (two buffers)', formula: 'log₂B', pages: 'pp. 260–261', sec: '11.3', secLabel: '11.3' },
  { what: 'Barriers in one block scan, Fig. 11.9 (warp scans)', formula: '2', pages: 'p. 265', sec: '11.4', secLabel: '11.4' },
  { what: 'Additions, Kogge-Stone on N elements', formula: 'N · log₂N − (N − 1)', pages: 'p. 266', sec: '11.5', secLabel: '11.5' },
  { what: 'Additions, coarsened, P threads', formula: '2N + P · log₂P − P − N/P, about 2N when P ≪ N', pages: 'pp. 269–270', sec: '11.6', secLabel: '11.6' },
  { what: 'Additions, Brent-Kung on N elements', formula: '2N − 2 − log₂N', pages: 'p. 284', sec: '11.10', secLabel: '11.10' },
  { what: 'Global traffic, three kernels: scan, scan, add', formula: '16 bytes per element', pages: 'p. 274', sec: '11.9', secLabel: '11.9' },
  { what: 'Global traffic, three kernels: reduce, scan, scan', formula: '12 bytes per element', pages: 'pp. 275–276', sec: '11.9', secLabel: '11.9' },
  { what: 'Global traffic, one pass: load once, store once', formula: '8 bytes per element', pages: 'p. 272', sec: '11.8', secLabel: '11.8, 11.9' },
];

export const questionsS110: Record<string, Question> = {
  'ch11/start/q-assoc': {
    kind: 'choice',
    id: 'ch11/start/q-assoc',
    prompt: (
      <>
        A reduction tree regroups the additions of a loop. It gives the loop's answer for every input when the operator
        is:
      </>
    ),
    options: [
      { label: 'associative', correct: true },
      { label: 'commutative', why: 'Commutativity lets you swap the two operands. A tree that keeps the operands in order only regroups them, and regrouping is what associativity allows.' },
      { label: 'one that has an identity', why: 'An identity tells idle threads what to contribute. It says nothing about whether regrouping changes the result.' },
      { label: 'any of these', why: 'Only one of the three is about regrouping.' },
    ],
    explain: (
      <p>
        A tree changes where the parentheses go, which is safe exactly when the operator is associative. Commutativity is
        needed only when the tree also changes the order of the operands, as the convergent kernel of chapter 10 does.
        Float addition is associative only approximately, so a tree changes the last bits of a float sum. Scan inherits
        exactly this requirement: each output is a reduction of a prefix (p. 255).
      </p>
    ),
    source: 'ch. 10 §10.2, p. 224; ch. 11 p. 255; float addition: notes/today/ch10-facts.md A9',
  },
  'ch11/start/q-shfl-edge': {
    kind: 'choice',
    id: 'ch11/start/q-shfl-edge',
    prompt: (
      <>
        In a warp of 32 lanes, lane 30 calls <code>__shfl_down_sync(0xffffffff, v, 4)</code>. What does it get back?
      </>
    ),
    options: [
      { label: "Lane 34's value", why: 'There is no lane 34 in a 32-lane warp.' },
      { label: 'Its own v', correct: true },
      { label: '0', why: 'Nothing is filled in. Zero would be the identity of +, but the shuffle does not know which operator you use.' },
      { label: "Lane 2's value (it wraps around)", why: 'That is what __shfl_sync does with an out-of-range lane, not the down shuffle.' },
    ],
    explain: (
      <p>
        NVIDIA documents that the source lane "will not wrap around", so the upper lanes "effectively remain unchanged":
        lane 30 receives its own value. This chapter uses the mirror image, <code>__shfl_up_sync</code>, whose lower
        lanes keep their own value. That is why its loop needs the test <code>laneIdx() &gt;= stride</code>: without it
        those lanes would add their own value to itself.
      </p>
    ),
    source: 'ch. 10 §10.7; notes/today/infra-facts.md A3; ch11-facts A3',
  },
  'ch11/start/q-false-dep': {
    kind: 'choice',
    id: 'ch11/start/q-false-dep',
    prompt: (
      <>
        A loop reads a neighbour's slot, waits at a barrier, writes its own slot, and waits again. You give it two
        buffers that swap roles every iteration. Which barrier can go?
      </>
    ),
    options: [
      { label: 'The one after the read', correct: true },
      { label: 'The one after the write', why: 'That one makes the next reader wait for the value it needs. Two buffers do not change that.' },
      { label: 'Both', why: 'One of them guards data that has to arrive before anyone reads it.' },
      { label: 'Neither', why: 'One of them exists only because the same slot is reused.' },
    ],
    explain: (
      <p>
        The barrier after the read stops a writer from destroying a value someone still has to read: a write-after-read,
        or false, dependence. It exists only because one slot holds both the old and the new value. With two buffers the
        new value goes elsewhere and that barrier has nothing to protect. The barrier after the write guards a true
        (read-after-write) dependence and stays. Section 11.3 does exactly this to the scan kernel.
      </p>
    ),
    source: 'ch. 5 §5.4, pp. 110–111; ch. 6 §6.7, pp. 146–147',
  },
  'ch11/start/q-sectors': {
    kind: 'numeric',
    id: 'ch11/start/q-sectors',
    prompt: (
      <>
        The 32 lanes of a warp each load the float at index <code>4*lane</code> of an aligned array, as they would if
        every thread loaded its own run of 4 elements directly. NVIDIA moves global memory in 32-byte sectors. How many
        sectors does this one load touch?
      </>
    ),
    answer: 16,
    unit: 'sectors',
    explain: (
      <p>
        Lane 31 reads element 124, bytes 496 to 499, so the warp spans bytes 0 to 499: sixteen 32-byte sectors, of which
        it uses 4 bytes in every 16. Consecutive indices would touch 4 sectors. Section 11.6 hits exactly this when each
        thread owns a run of elements, and loads chunk by chunk to avoid it (p. 269).
      </p>
    ),
    source: "ch. 6 §6.1, pp. 124–132; the 32-byte sector is NVIDIA's (notes/today/ch06-facts.md A2); ch. 11 p. 269",
  },
  'ch11/start/q-block-order': {
    kind: 'choice',
    id: 'ch11/start/q-block-order',
    prompt: <>In a grid of many blocks, block 7 spins in a loop until block 6 sets a flag in global memory. Is that safe?</>,
    options: [
      {
        label: 'No: nothing promises block 6 has been scheduled, and if every resident block is spinning it never will be.',
        correct: true,
      },
      { label: 'Yes: blocks start in order of blockIdx.x.', why: 'NVIDIA says the scheduler makes no ordering guarantees.' },
      { label: 'Yes, as long as the flag is read with an atomic.', why: 'An atomic makes the read well defined; it does not make block 6 run.' },
      { label: 'Only if both blocks are in the same warp.', why: 'Blocks are never in the same warp.' },
    ],
    explain: (
      <p>
        Blocks may run in any order and need not be resident together. If the device is full of blocks that all wait for
        lower-numbered blocks that have not started, nothing frees a slot and the grid hangs. Section 11.9 needs
        exactly this kind of waiting for its single-pass scan, and fixes the order by handing out block indices as blocks
        actually start.
      </p>
    ),
    source: 'ch. 4 §4.3, p. 73; notes/today/ch04-facts.md A5; ch. 11 p. 277',
  },
  'ch11/start/q-relaxed': {
    kind: 'choice',
    id: 'ch11/start/q-relaxed',
    prompt: (
      <>
        Why was <code>cuda::memory_order_relaxed</code> enough for the histogram's <code>fetch_add</code> in chapter 9?
      </>
    ),
    options: [
      {
        label: "Nothing the thread does depends on where the atomic lands relative to its other memory accesses; its one other access, the load that picks the bin, is already ordered by the data dependence.",
        correct: true,
      },
      { label: 'Relaxed is the fastest, and correctness is the same for every order.', why: 'Orders differ in what they promise about the other accesses around the atomic. Chapter 11 needs a stronger one.' },
      { label: 'Histogram bins are in shared memory, where order does not matter.', why: 'The basic kernel updates bins in global memory, and the order question is about other accesses, not where the bin lives.' },
      { label: 'Atomics are always sequentially consistent on GPUs.', why: 'The order is a parameter precisely because they are not, unless you ask.' },
    ],
    explain: (
      <p>
        A memory order constrains how an atomic may be reordered with the same thread's other accesses. The histogram
        thread loads a pixel, and that value chooses the bin, so the load must come first anyway. In section 11.9 a block
        publishes a sum and then sets a flag, and another block reads the flag and then the sum. There the order matters,
        and the book switches to release and acquire.
      </p>
    ),
    source: 'ch. 9 §9.2, pp. 208–209; ch. 11 pp. 278–279',
  },
};

export function S110() {
  const q = questionsS110;
  return (
    <>
      <p>
        A scan is a running total. Output <i>i</i> is the sum of the inputs up to <i>i</i>, so every output seems to
        need the one before it: about as sequential as a computation can look. The chapter shows how many threads can
        compute it anyway, and what that costs. The book's rule of thumb is broad: when each item of a computation is
        defined from the item before it, the computation can often be written as a scan and run in parallel.
      </p>
      <Source pages="251" />
      <p>
        Scan is also a building block. Filtering (chapter 12) and radix sort (chapter 14) are built on it, and the book
        lists quick sort, string comparison, polynomial evaluation, recurrences and tree operations as further users.
        And it is a study in a trade-off you have not met in this form yet. The fast parallel scans do more additions in
        total than the plain loop; on hardware with few execution units, that extra work can make them slower. The
        chapter keeps both numbers in view: the total work and the number of steps (the span).
      </p>
      <Source pages="251–252" />

      <h2>Why you will care</h2>
      <p>
        Top-p sampling sorts the next token's probabilities and keeps the shortest prefix whose cumulative probability
        reaches p. That cumulative probability is a scan, and whether the token that crosses p is kept is the difference
        between the two kinds of scan defined in section 11.1. Lab 11.6 builds that keep mask; it is the first part of
        milestone M5 of the course project. Scans also turn sizes into offsets: the per-sequence token counts of a
        batch become the positions where each sequence starts in one packed buffer, and in chapter 12 a scan of 0/1 keep
        flags gives every kept token its slot in a compacted list.
      </p>
      <Beyond>
        <p>
          The book never connects scan to sampling or batching. These uses are ours (notes, "Relevance to LLM-inference
          kernels").
        </p>
      </Beyond>

      <h2>What runs under torch.cumsum today</h2>
      <p>
        Call <code>torch.cumsum(probs, dim=-1)</code> on a batch of rows on a CUDA device and PyTorch scans each row along
        its last dimension with a kernel of its own: several rows per block, a shared-memory scan per chunk, and a
        running total carried from chunk to chunk. A one-dimensional tensor goes to NVIDIA's CUB library instead, unless
        you have asked for deterministic algorithms. On Apple GPUs (MPS), long rows are split across threadgroups and
        scanned in three passes: reduce each block, scan the block sums, add them back. You will recognise every one of
        these pieces by the end of the chapter.
      </p>
      <Today checked="2026-11-20 (ch11-facts D4)">
        <p>
          CUDA: <code>scan_innermost_dim</code> for the last dimension, "Parallel reduction with Sklansky method" inside
          a block; 1-D tensors through CUB, or a deterministic two-kernel scan under{' '}
          <code>use_deterministic_algorithms</code>. MPS: "Three-pass scan (block_reduce -&gt; scan_block_sums -&gt;
          block_carry)".
        </p>
      </Today>

      <h2>The chapter as a cost ledger</h2>
      <p>
        The book builds one kernel and then removes its costs one at a time: barriers, extra additions, then trips to
        global memory. Here are the numbers it will give, so you can watch each one fall. B is the number of elements one
        block scans, N the length of the whole input, P the number of threads.
      </p>
      <table className="data wide" data-part="ledger" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr>
            <th>What it costs</th>
            <th>Formula</th>
            <th>Book</th>
            <th>Section</th>
          </tr>
        </thead>
        <tbody>
          {LEDGER.map((r) => (
            <tr key={r.what}>
              <td>{r.what}</td>
              <td className="nowrap">{r.formula}</td>
              <td className="nowrap">{r.pages}</td>
              <td>
                <a href={href('ch11', r.sec)}>{r.secLabel}</a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>
        Two rows deserve a first look now. Kogge-Stone, the parallel scan this chapter starts from, does about log₂N
        times the additions of the loop: for 512 elements between 8 and 9 times as many (p. 266). Coarsening brings that
        back to about two additions per element. And the traffic rows say that the best global scan moves 8 bytes per
        element, half of what the obvious three-kernel design moves.
      </p>
      <Aside>
        <p>
          The barrier counts are the loops' own arithmetic: Fig. 11.3 runs log₂B iterations with two barriers each, Fig.
          11.5 one each (pp. 257, 261).
        </p>
      </Aside>

      <h2>Before you start</h2>
      <p>
        Six questions on what this chapter assumes from chapters 4 to 10. Of chapter 10, the reduction trees (
        <a href={href('ch10', '10.2')}>10.2</a>), warp shuffles (<a href={href('ch10', '10.7')}>10.7</a>,{' '}
        <a href={href('ch10', '10.8')}>10.8</a>) and coarsening (<a href={href('ch10', '10.10')}>10.10</a>) are the pages
        this chapter leans on most. Each question comes back in <a href={href('review')}>review</a>.
      </p>
      <Ask q={q['ch11/start/q-assoc']} label="From chapter 10" />
      <Ask q={q['ch11/start/q-shfl-edge']} label="From chapter 10" />
      <Ask q={q['ch11/start/q-false-dep']} label="From chapters 5 and 6" />
      <Ask q={q['ch11/start/q-sectors']} label="From chapter 6" />
      <Ask q={q['ch11/start/q-block-order']} label="From chapter 4" />
      <Ask q={q['ch11/start/q-relaxed']} label="From chapter 9" />

      <h2>The labs</h2>
      <p>
        Six labs follow the sections, all on the <a href={href('ch11', 'labs')}>labs page</a>. Each asks for a prediction
        before you measure, and each tests against a CPU reference computed with the plain left-to-right loop.
      </p>
      <p>
        <b>Lab 11.1, one block scan, three ways.</b> The two-barrier kernel of section 11.2, the double-buffered kernel
        of section 11.3 and the warp-scan kernel of section 11.4, on one block, measured against each other.
      </p>
      <p>
        <b>Lab 11.2, coarsening and register tiling.</b> Each thread first scans its own run of elements, sequentially
        and in registers (sections 11.6 and 11.7), so that the block-level scan has far fewer values to combine.
      </p>
      <p>
        <b>Lab 11.3, scan race hunt.</b> A row-wise scan in the shape llama.cpp, MLX and PyTorch ship, with three
        defects to find and fix. The emulators show some that a GPU can hide; the fixed kernel is the core of lab 11.6.
      </p>
      <p>
        <b>Lab 11.4, a scan of any length.</b> Block scans stitched into a scan of the whole input with separate
        kernels (section 11.9), the design that works on every backend.
      </p>
      <p>
        <b>Lab 11.5, single-pass scan with lookback.</b> Optional, and for CUDA and HIP only: blocks wait for each other
        through flags in global memory (section 11.9). Metal documents no guarantee that makes this safe.
      </p>
      <p>
        <b>Lab 11.6, top-p keep mask.</b> Milestone M5, part 1: for each row of sorted probabilities, a scan and a
        threshold that keep the tokens whose mass before them is below p. It states its tie rule and the rounding band
        around p inside which either answer is accepted.
      </p>
      <Today checked="2026-11-20 (ch11-facts C5, B3)">
        <p>
          Metal's spec gives no forward-progress guarantee between threadgroups, and an Apple engineer called a spinning
          chained scan "undefined behavior". AMD documents none either; the HIP version stays unverified until it runs on
          AMD hardware.
        </p>
      </Today>
    </>
  );
}
