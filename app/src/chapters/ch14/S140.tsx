import { Ask, type Question } from '../../components/Ask';
import { Beyond, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';

// Before you start. Prerequisite questions from the notes of chapters 4, 6, 10, 11, 12 and 13;
// vendor and project facts from notes/today/ch14-facts.md (A1, A7, A9, D3, D7), ch06-facts.md
// (A2) and ch10-facts.md (A1).
export const questionsS140: Record<string, Question> = {
  'ch14/start/q-exscan': {
    kind: 'numeric',
    id: 'ch14/start/q-exscan',
    prompt: (
      <>
        Take the exclusive scan (with +) of the bits <code>[1, 0, 1, 1, 0, 0, 1, 1]</code>. What is the value at index
        5?
      </>
    ),
    answer: 3,
    explain: (
      <p>
        An exclusive scan puts at each index the sum of everything strictly before it: 1 + 0 + 1 + 1 + 0 = 3. When the
        inputs are bits, that sum is the number of ones before each position. Section{' '}
        <a href={href('ch14', '14.5')}>14.5</a> needs exactly this number for every key of a radix pass, and calls it
        "ones before". Back to <a href={href('ch11', '11.1')}>section 11.1</a>.
      </p>
    ),
    source: 'ch. 11 §11.1, p. 253',
  },
  'ch14/start/q-filterpos': {
    kind: 'numeric',
    id: 'ch14/start/q-filterpos',
    prompt: (
      <>
        A stable filter keeps the elements whose flags are <code>[0, 1, 1, 0, 1]</code>. At which output position does
        input element 4 land?
      </>
    ),
    answer: 2,
    explain: (
      <p>
        The exclusive scan of the flags is <code>[0, 0, 1, 2, 2]</code>: element 4 has two kept elements before it, so
        it takes slot 2, and the kept elements stay in input order. A radix pass is this filter with a second output
        list for the elements that fail, also in input order (section <a href={href('ch14', '14.4')}>14.4</a>). Back
        to <a href={href('ch12', '12.5')}>section 12.5</a>.
      </p>
    ),
    source: 'ch. 12 §12.5, pp. 297–298',
  },
  'ch14/start/q-stride-sectors': {
    kind: 'numeric',
    id: 'ch14/start/q-stride-sectors',
    prompt: (
      <>
        The 32 lanes of a warp store 4-byte values. Lane <i>k</i> writes element <code>2k</code> of an array whose
        element 0 sits on a 32-byte boundary. How many 32-byte sectors does the store touch?
      </>
    ),
    answer: 8,
    unit: 'sectors',
    explain: (
      <p>
        Lane 31 writes element 62, bytes 248 to 251, so the store spans bytes 0 to 251: eight aligned 32-byte sectors,
        twice the four that 32 contiguous values need, and every sector is half wasted. NVIDIA serves global memory in
        these 32-byte units. Section <a href={href('ch14', '14.6')}>14.6</a> counts the cost of a radix pass's
        scattered writes the same way. Back to <a href={href('ch06', '6.1')}>section 6.1</a>.
      </p>
    ),
    source: 'ch. 6 §6.1, pp. 124–132; the 32-byte sector: notes/today/ch06-facts.md A2',
  },
  'ch14/start/q-flagrace': {
    kind: 'choice',
    id: 'ch14/start/q-flagrace',
    prompt: (
      <>
        Many threads of a kernel write the value 1 to the same flag in global memory, with plain stores and nothing to
        order them. What is this?
      </>
    ),
    options: [
      {
        label: 'Not a race: writes that all store the same value cannot conflict, so the flag ends up 1 either way.',
        why: 'Idempotent is not the same as race-free. The memory model defines a race by the accesses, not by the values.',
      },
      {
        label: 'Only a problem if the threads wrote different values; equal stores are idempotent, so the result is defined.',
        why: 'With different values the result would be unpredictable as well as undefined. With equal values it is still undefined.',
      },
      {
        label: 'Undefined on AMD GPUs, but NVIDIA\'s memory model treats equal-value stores as benign.',
        why: 'The rule is NVIDIA\'s own: any data race in the CUDA memory model is undefined behaviour.',
      },
      {
        label: 'A data race: undefined behaviour in the CUDA memory model although every write stores 1. Make the store atomic.',
        correct: true,
      },
    ],
    explain: (
      <p>
        The book calls this race benign in practice, then warns that the C++ memory model makes a data race undefined
        even when the operation is idempotent, and advises an atomic (p. 333). NVIDIA's current guide says the same of
        CUDA. Odd-even sort's <code>hasChanged</code> flag is this case (section{' '}
        <a href={href('ch14', '14.2')}>14.2</a>). Back to <a href={href('ch10', '10.3')}>section 10.3</a>, where the
        memory model first comes up.
      </p>
    ),
    source: 'ch. 14 §14.2, p. 333; ch. 10 §10.3, p. 230; memory model: notes/today/ch10-facts.md A1, ch14-facts A9',
  },
  'ch14/start/q-gridscan': {
    kind: 'choice',
    id: 'ch14/start/q-gridscan',
    prompt: (
      <>
        A kernel launched the ordinary way wants to run a scan over the whole grid in its middle, with every thread
        waiting for the result before it goes on. Why can it not?
      </>
    ),
    options: [
      {
        label: 'Shared memory is private to a block, so no block can see the bits the other blocks extracted.',
        why: 'Shared memory is private, but the bits can go through global memory, which every block sees. The obstacle is waiting for blocks that may not be running.',
      },
      {
        label: 'Blocks need not all be resident or run in any order, so a block waiting for the rest may wait forever.',
        correct: true,
      },
      {
        label: 'It could: __syncthreads() waits for every thread of the launch, so the scan would only be slower, never wrong.',
        why: '__syncthreads() spans one block only, and a hand-made wait across blocks may never end. Speed is not the issue.',
      },
    ],
    explain: (
      <p>
        A barrier only spans a block, and the run time may run blocks in any order, a few at a time. Chapter 11 crosses
        blocks either with separate launches, or with a cooperative launch that keeps every block resident (on CUDA
        and HIP), or with flags that only wait for earlier blocks. The book's radix kernel calls a grid-wide scan in
        its middle; section <a href={href('ch14', '14.5')}>14.5</a> shows what has to change. Back to{' '}
        <a href={href('ch04', '4.3')}>section 4.3</a> and <a href={href('ch11', '11.9')}>section 11.9</a>.
      </p>
    ),
    source: 'ch. 4 §4.3, pp. 70–74; ch. 11 §11.9, pp. 273–277',
  },
  'ch14/start/q-corank': {
    kind: 'choice',
    id: 'ch14/start/q-corank',
    prompt: (
      <>
        A thread must write outputs <code>k</code> to <code>k + 7</code> of the merge of two sorted lists, without
        merging anything before <code>k</code>. What does it need first?
      </>
    ),
    options: [
      {
        label: 'The value of output k − 1, so that it can carry on merging from that value in both lists.',
        why: 'Knowing a value does not say where in each input to start. It also needs the merge of everything before it.',
      },
      {
        label: 'Element k of each input list, since the first k outputs are the k smallest of the two lists together.',
        why: 'The first k outputs can come from the two lists in any proportion; k of each is right only by accident.',
      },
      {
        label: 'The co-rank pair (i, j), i + j = k: how many of the first k outputs each list supplies, by binary search.',
        correct: true,
      },
    ],
    explain: (
      <p>
        The co-rank function turns an output position into the two input positions that feed it; after that each
        thread runs an ordinary sequential merge on its slices. Merge sort repeats chapter 13's merge at every level of
        its tree (section <a href={href('ch14', '14.3')}>14.3</a>). Back to{' '}
        <a href={href('ch13', '13.3')}>section 13.3</a>.
      </p>
    ),
    source: 'ch. 13 §13.3–13.4, pp. 306–313',
  },
};

const SECTIONS: { id: string; what: string }[] = [
  { id: '14.1', what: 'what a sort must deliver, stable and unstable sorts, comparison and non-comparison sorts' },
  { id: '14.2', what: 'odd-even sort: neighbouring pairs in parallel, alternating; simple and slow' },
  { id: '14.3', what: 'merge sort: chapter 13\'s merge, run at every level of a tree' },
  { id: '14.4', what: 'radix sort one digit at a time, and why every pass must be stable; float logits as keys (ours)' },
  { id: '14.5', what: 'one radix pass in parallel: an exclusive scan of the bits, then a scatter' },
  { id: '14.6', what: 'sort locally in shared memory first, so the writes to global memory coalesce' },
  { id: '14.7', what: 'wider digits: fewer passes, worse coalescing, a bigger scan' },
  { id: '14.8', what: 'several keys per thread, for bigger local buckets and a smaller scan' },
  { id: '14.9', what: 'sorting networks, sample sort, and most-significant-digit radix sort' },
];

export function S140() {
  const q = questionsS140;
  return (
    <>
      <p>
        Sorting puts a list in order. The book's reason to care is general: ordered data is cheap to search, cheap to
        compare with other ordered data, and many algorithms run faster on it. Efficient sorts are also hard to
        parallelise, which makes them a good case study in the techniques of the last few chapters.
      </p>
      <Source pages="329" />

      <h2>Where it sits under a sampler</h2>
      <p>
        You have probably run a GPU sort without writing one. A top-p (nucleus) sampler keeps the smallest set of most
        likely tokens whose probabilities add up to at least p. The usual way to find that set orders the vocabulary
        by probability, takes a cumulative sum, and cuts where the sum crosses p: a sort, then chapter 11's scan.
      </p>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md, D3, D7)">
        <p>
          Hugging Face's <code>TopPLogitsWarper</code>: <code>torch.sort</code>, softmax, <code>cumsum</code>.
          llama.cpp's experimental backend top-p: <code>ggml_argsort</code> descending, <code>ggml_soft_max</code>,{' '}
          <code>ggml_cumsum</code>.
        </p>
      </Today>
      <p>
        Two qualifications. Top-k needs no full sort: finding the k largest is a selection, which is why PyTorch has{' '}
        <code>torch.topk</code> and CUB now has <code>cub::DeviceTopK</code>; module X1 builds one in{' '}
        <a href={href('x1', 'X1.4')}>X1.4, "Top-k is a selection, not a sort"</a>. And top-p over
        a sorted list is the common formulation, not the only one: vLLM's default Triton path finds the cut with a pivot
        search and never sorts.
      </p>
      <Beyond>
        <p>
          The book does not mention sampling in this chapter; this bridge is ours, from the project sources in
          ch14-facts A7 and D7.
        </p>
      </Beyond>

      <h2>What the chapter builds</h2>
      <p>
        Three sorts, weighted unevenly. Odd-even sort is brief because it is simple. Merge sort is brief because its
        hard part is chapter 13's parallel merge. Radix sort takes the rest of the chapter: it compares no keys at
        all, and its optimisations are a more general form of chapter 12's stable filter.
      </p>
      <Source pages="329" />
      <table className="data" data-part="sections">
        <thead>
          <tr>
            <th>Section</th>
            <th>What you do there</th>
          </tr>
        </thead>
        <tbody>
          {SECTIONS.map((s) => (
            <tr key={s.id}>
              <td className="nowrap">
                <a href={href('ch14', s.id)}>{s.id}</a>
              </td>
              <td>{s.what}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>
        The kernel the chapter takes apart is its Fig. 14.7: one pass of a radix sort, which is an exclusive scan of
        one bit per key followed by a scatter of every key to its own slot. Sections 14.6 to 14.8 then make its writes
        coalesce.
      </p>
      <Source section="14.5" pages="339" figs="Fig. 14.7" />
      <p>
        The project piece is milestone M5's sorted candidates: lab 14.3 sorts every row of a batch of probabilities
        in descending order, stably, with the token ids carried along. The <a href={href('ch14', 'labs')}>labs page</a>{' '}
        has it and two smaller labs that lead up to it.
      </p>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md, A1)">
        <p>
          The book ends by advising Thrust or CUB over a sort of your own (p. 347), and that stands. CUB's radix sort
          today is the OneSweep algorithm the book cites on p. 336: its default for keys of 4 bytes or more from
          compute capability 6.0, and for every key size from 9.0.
        </p>
      </Today>

      <h2>Before you start</h2>
      <p>
        Six questions on what this chapter assumes from chapters 4, 6 and 10 to 13. Each explanation names the section
        of this chapter that relies on it, and each question returns in <a href={href('review')}>review</a>. This
        course's order (section <a href={href('ch01', '1.7')}>1.7</a>) reads chapter 13 after this one, so if you follow
        it, take the co-rank question as a preview: section 14.3 uses only the search's result, and{' '}
        <a href={href('ch13', '13.3')}>section 13.3</a> works it out.
      </p>
      <Ask q={q['ch14/start/q-exscan']} label="From chapter 11" />
      <Ask q={q['ch14/start/q-filterpos']} label="From chapter 12" />
      <Ask q={q['ch14/start/q-stride-sectors']} label="From chapter 6" />
      <Ask q={q['ch14/start/q-flagrace']} label="From chapter 10" />
      <Ask q={q['ch14/start/q-gridscan']} label="From chapters 4 and 11" />
      <Ask q={q['ch14/start/q-corank']} label="From chapter 13" />
    </>
  );
}
