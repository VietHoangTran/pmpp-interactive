import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  TIES16,
  TIES8,
  applyNetwork,
  bitonicCount,
  bitonicStages,
  byValue,
  tieGroups,
  transpositionCount,
  type Item,
} from '../../sim/sortnet';

const fmtP = (p: number) => (Number.isInteger(p) ? String(p) : p.toFixed(3).replace(/0+$/, '').replace(/\.$/, ''));

/** Reads exactly n non-negative numbers; null if malformed. */
function parseKeys(text: string, n: number): number[] | null {
  const parts = text.split(/[\s,]+/).filter(Boolean);
  if (parts.length !== n) return null;
  const xs = parts.map(Number);
  return xs.every((v) => Number.isFinite(v) && v >= 0) ? xs : null;
}

/** A deterministic shuffle, so a check can predict it: the k-th shuffle of the preset. */
function shuffled<T>(xs: T[], seed: number): T[] {
  const out = xs.slice();
  let s = seed * 2654435761 >>> 0;
  for (let i = out.length - 1; i > 0; i--) {
    s = (s * 1103515245 + 12345) >>> 0;
    const j = (s >>> 8) % (i + 1);
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

function Network() {
  const [n, setN] = useState<8 | 16>(8);
  const [keys, setKeys] = useState<number[]>(TIES8.map((x) => x.key));
  const [text, setText] = useState(TIES8.map((x) => fmtP(x.key)).join(' '));
  const [byIndex, setByIndex] = useState(false);
  const [done, setDone] = useState(0); // stages applied
  const [shuffles, setShuffles] = useState(0);
  const stages = useMemo(() => bitonicStages(n), [n]);
  const items: Item[] = keys.map((key, id) => ({ key, id }));
  const run = useMemo(() => applyNetwork(items, stages, byValue(byIndex)), [keys, stages, byIndex]); // eslint-disable-line react-hooks/exhaustive-deps
  const at = Math.min(done, stages.length);
  const cur = at === 0 ? items : run.stages[at - 1].after;
  const finished = at === stages.length;
  const groups = finished ? tieGroups(cur) : [];
  const bad = parseKeys(text, n) === null;

  const load = (m: 8 | 16) => {
    const preset = m === 8 ? TIES8 : TIES16;
    setN(m);
    setKeys(preset.map((x) => x.key));
    setText(preset.map((x) => fmtP(x.key)).join(' '));
    setDone(0);
    setShuffles(0);
  };
  const shuffle = () => {
    const k = shuffles + 1;
    const next = shuffled(keys, k);
    setShuffles(k);
    setKeys(next);
    setText(next.map(fmtP).join(' '));
    setDone(0);
  };

  // Geometry.
  const gap = n === 8 ? 34 : 25;
  const top = 30;
  const left = 96;
  const colW = n === 8 ? 82 : 52;
  const width = left + stages.length * colW + 100;
  const height = top + n * gap + 10;
  const wy = (k: number) => top + k * gap;
  const sx = (s: number) => left + s * colW + colW / 2;
  const swappedNow = at > 0 ? run.stages[at - 1].swapped : [];

  return (
    <Bench
      title="Bitonic network"
      task="Step through the stages, shuffle the values and step again, then tick the index comparison and run the ties preset."
      foot={
        <>
          Each vertical line is a comparator: it looks at two positions and sends the larger value to the arrow's end,
          swapping only when the pair is strictly out of order. Sorted largest first. The small number beside each value
          is its token index, its position in the input.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label="N">
          <button aria-pressed={n === 8} onClick={() => load(8)}>N = 8</button>
          <button aria-pressed={n === 16} onClick={() => load(16)}>N = 16</button>
        </span>
        <button onClick={() => load(n)}>ties preset</button>
        <button onClick={shuffle}>shuffle values</button>
        <label className="check">
          <input type="checkbox" checked={byIndex} onChange={(e) => { setByIndex(e.target.checked); setDone(0); }} /> compare (value, index)
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field" style={{ flex: '1 1 20rem' }}>
          Values ({n}, separated by spaces)
          <input type="text" value={text} aria-label="values" style={{ width: '100%' }}
            onChange={(e) => { setText(e.target.value); const v = parseKeys(e.target.value, n); if (v) { setKeys(v); setDone(0); } }} />
        </label>
        {bad ? <span className="chip fail">exactly {n} numbers, none negative</span> : null}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setDone(0)} disabled={at === 0}>Restart</button>
        <button onClick={() => setDone(at - 1)} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => setDone(at + 1)} disabled={finished}>Step</button>
        <button onClick={() => setDone(stages.length)} disabled={finished}>Run</button>
        <span className="muted" aria-live="polite" data-stage={at}>stage {at} of {stages.length}</span>
      </div>
      <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Bitonic sorting network" style={{ width: '100%', maxWidth: width, minWidth: 600, display: 'block' }}
        data-values={cur.map((x) => `${fmtP(x.key)}:${x.id}`).join(' ')}>
        {stages.map((_, s) => (
          <rect key={'col' + s} x={left + s * colW + 2} y={top - 18} width={colW - 4} height={n * gap + 4} rx="3"
            fill={s === at - 1 ? 'var(--rule-soft)' : 'none'} />
        ))}
        {stages.map((_, s) => (
          <text key={'h' + s} x={sx(s)} y={top - 22} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{s + 1}</text>
        ))}
        {Array.from({ length: n }, (_, k) => (
          <g key={'w' + k}>
            <line x1={left - 4} y1={wy(k)} x2={left + stages.length * colW + 4} y2={wy(k)} stroke="var(--rule)" />
            <text x={left - 10} y={wy(k) + 4} fontSize="12" textAnchor="end" data-in={k}>
              {fmtP(items[k].key)}<tspan fontSize="9" style={{ fill: 'var(--ink-3)' }}> t{items[k].id}</tspan>
            </text>
            <text x={left + stages.length * colW + 12} y={wy(k) + 4} fontSize="12" data-out={k}
              style={{ fill: finished && groups.some((g) => g.key === cur[k].key) ? (groups.find((g) => g.key === cur[k].key)!.increasing ? 'var(--ok)' : 'var(--fail)') : 'var(--ink)' }}>
              {fmtP(cur[k].key)}<tspan fontSize="9" style={{ fill: 'var(--ink-3)' }}> t{cur[k].id}</tspan>
            </text>
          </g>
        ))}
        {stages.map((stage, s) =>
          stage.map((c, j) => {
            // Comparators of one stage that overlap vertically get their own slots across the column.
            const d = c.hi - c.lo;
            const slot = c.lo % d;
            const x = d === 1 ? sx(s) : left + s * colW + 6 + ((slot + 0.5) * (colW - 12)) / d;
            const y1 = wy(c.lo);
            const y2 = wy(c.hi);
            const live = s === at - 1;
            const swapped = live && swappedNow[j];
            const tipY = c.up ? y1 : y2; // the larger value goes to the tip
            const dir = c.up ? 1 : -1;
            return (
              <g key={`${s}-${j}`} opacity={s < at ? 1 : 0.45} data-comp={`${s}:${c.lo}-${c.hi}`} data-swapped={swapped ? 'yes' : 'no'}>
                <line x1={x} y1={y1} x2={x} y2={y2} stroke="var(--ink)" strokeWidth={swapped ? 3 : 1.3} />
                <circle cx={x} cy={c.up ? y2 : y1} r={2.6} fill="var(--ink)" />
                <path d={`M${x - 4} ${tipY + dir * 7} L${x} ${tipY} L${x + 4} ${tipY + dir * 7} Z`} fill="var(--ink)" />
              </g>
            );
          }),
        )}
      </svg>
      </div>
      <table className="threads" aria-label="Counters" style={{ marginTop: '0.8rem' }}>
        <tbody>
          <tr>
            <td style={{ whiteSpace: 'normal' }}>comparators applied</td>
            <td style={{ whiteSpace: 'normal' }}><b data-counter="applied">{(at * n) / 2}</b> of {bitonicCount(n)} ({stages.length} stages of {n / 2})</td>
          </tr>
          <tr>
            <td style={{ whiteSpace: 'normal' }}>swaps in the last stage</td>
            <td><b data-counter="swaps">{swappedNow.filter(Boolean).length}</b></td>
          </tr>
          <tr>
            <td style={{ whiteSpace: 'normal' }}>odd-even transposition, same N</td>
            <td style={{ whiteSpace: 'normal' }}><b data-counter="transposition">{transpositionCount(n)}</b> comparators ({n} steps)</td>
          </tr>
        </tbody>
      </table>
      {finished ? (
        <div style={{ marginTop: '0.8rem' }} data-ties={groups.length === 0 ? 'none' : groups.every((g) => g.increasing) ? 'stable' : 'reordered'}>
          {groups.length === 0 ? (
            <span className="muted">No ties in these values.</span>
          ) : (
            groups.map((g) => (
              <span key={g.key} className={'chip ' + (g.increasing ? 'ok' : 'fail')} style={{ marginRight: '0.4rem' }} data-group={fmtP(g.key)} data-ok={g.increasing ? 'yes' : 'no'}>
                {fmtP(g.key)}: {g.ids.map((i) => 't' + i).join(' ')}
              </span>
            ))
          )}
        </div>
      ) : null}
    </Bench>
  );
}

export const questionsS149: Record<string, Question> = {
  'ch14/14.9/q-comparators': {
    kind: 'numeric',
    id: 'ch14/14.9/q-comparators',
    prompt: <>A bitonic sorting network for N = 16 has how many comparators?</>,
    answer: 80,
    explain: (
      <p>
        log2 16 = 4, so the network has 4 × 5 / 2 = 10 stages, each with N/2 = 8 comparators: 80. Odd-even transposition
        sort, run as a fixed network for 16 steps, needs 8 steps of 8 and 8 of 7: 120. For N = 1,024 the gap is 28,160
        against about half a million.
      </p>
    ),
    source: 'ch. 14 §14.9, p. 345; src/sim/sortnet.ts',
  },
  'ch14/14.9/q-network-ties': {
    kind: 'choice',
    id: 'ch14/14.9/q-network-ties',
    prompt: (
      <>
        A sampler sorts 64 candidates per row inside one block and must list equal probabilities by increasing token
        index. What do you write?
      </>
    ),
    options: [
      { label: 'A one-bit radix sort over global memory.', why: 'It would be stable, but 32 passes through global memory for 64 values is a great deal of traffic for a job one block can do on chip.' },
      { label: 'Odd-even sort with a host loop.', why: 'Stable enough here, but one launch and one copy back per step, for up to 64 steps per row.' },
      { label: 'A bitonic network comparing with >= instead of >.', why: 'Comparators reach across the array and carry other values past equal ones; changing the comparison inside one comparator does not make the network stable.' },
      {
        label: 'A bitonic network in shared memory or registers that compares (value, index) pairs.',
        correct: true,
      },
    ],
    explain: (
      <p>
        With (value, index) as the key, no two keys are equal, so the network's instability has nothing to act on: the output
        is the stable order, whatever path each value takes. 64 values fit easily in one block, and the fixed pattern
        needs no flag and no host loop.
      </p>
    ),
    source: 'ch. 14 §14.9, pp. 345–346; notes/today/ch14-facts.md D1',
  },
  'ch14/14.9/q-msd': {
    kind: 'choice',
    id: 'ch14/14.9/q-msd',
    prompt: <>Why does MSD radix suit top-k when LSD radix does not?</>,
    options: [
      {
        label: 'LSD decides the most significant digit last, so its early passes say nothing about which keys are largest. MSD\'s first pass already isolates the bucket holding the k-th largest key, and only that bucket needs more work.',
        correct: true,
      },
      { label: 'MSD makes fewer passes in total.', why: 'For a full sort both read every digit; the difference is what each pass tells you.' },
      { label: 'LSD is not stable.', why: 'LSD radix sort is stable; it has to be, for its passes to compose (section 14.4).' },
    ],
    explain: (
      <p>
        After an MSD pass on the top digit, every bucket above the one holding the k-th largest key is wholly in the top k
        and every bucket below it is wholly out; only one bucket remains undecided. That is why llama.cpp's top-k and
        PyTorch's multi-block top-k (<code>mbtopk</code>) count digit histograms from the most significant digit down.
      </p>
    ),
    source: 'ch. 14 §14.9, p. 346; notes/today/ch14-facts.md D2, D5',
  },
  'ch14/14.9/q-oversample': {
    kind: 'choice',
    id: 'ch14/14.9/q-oversample',
    prompt: (
      <>
        Sample sort cuts a million random keys into 4 buckets with 3 splitters, and each bucket is then sorted on its own.
        Version A uses 3 keys drawn at random as the splitters. Version B draws 30 keys, sorts them, and takes evenly spaced
        ones (the 8th, 15th and 23rd). Over many runs, what do you expect of the largest bucket?
      </>
    ),
    options: [
      {
        label: 'About a quarter of the input in both: the keys are random, so the splitters are spread evenly.',
        why: 'Random splitters are spread at random, not evenly. Three of them are often bunched, which leaves one wide gap.',
      },
      {
        label: 'Larger in B: more samples mean more chances of an unlucky draw.',
        why: 'Each splitter of B is the 8th, 15th or 23rd of 30 sorted samples, and such a rank varies far less than one random key.',
      },
      {
        label: 'In A it is often near half the input; in B it stays much nearer a quarter, so no bucket\'s sort holds up the rest.',
        correct: true,
      },
      {
        label: 'The same in A and B: on average every bucket is a quarter either way.',
        why: 'The average bucket is a quarter in both. The time waits for the largest bucket, and that is what differs.',
      },
    ],
    explain: (
      <p>
        Three random points cut the key range into four gaps, and the largest of them averages (1 + 1/2 + 1/3 + 1/4) / 4 =
        25/48 of the range: in version A one bucket typically holds about half the keys, and its sort is the critical path.
        Evenly spaced keys of a larger sorted sample vary much less, so the largest bucket shrinks towards a quarter as the
        sample grows. That is the book's "modest oversampling gives buckets of nearly equal size with high probability".
      </p>
    ),
    source: 'ch. 14 §14.9, p. 346 (the book cites Blelloch et al.; not checked by us); the 25/48 is our arithmetic',
  },
};

export function S149() {
  return (
    <>
      <p>
        The chapter ends with a survey. Its first point looks back at odd-even sort: the comparisons it makes follow a fixed
        pattern, the same whatever the data, and it parallelises easily because each step compares pairs that do not
        overlap. A whole family of methods works this way. They are called <b>sorting networks</b>, and the best known for
        parallel machines are Batcher's bitonic sort and odd-even merge sort.
      </p>
      <Source section="14.9" pages="345–346" />
      <p>
        They sort a fixed number of values with O(N log² N) comparators: far fewer than odd-even transposition's O(N²), more
        than the O(N log N) of merge sort. Asymptotically worse, then, yet often the fastest choice for short sequences,
        because they are simple. On a GPU that simplicity has a concrete form: the pattern does not depend on the data, so
        there is no flag, no host loop and no search, and every thread knows its partner at every stage.
      </p>
      <Beyond>
        <p>
          The book says "simplicity"; the list of what that buys on a GPU is ours.
        </p>
      </Beyond>
      <p>
        Bitonic sort is not stable. Its comparators reach across the array, so a value can be carried past an equal value
        by comparisons with others. Comparing (value, index) pairs removes the problem: no two keys are then equal, and the
        result is the stable order.
      </p>
      <Simplification>
        <p>
          The network is drawn for N a power of two only; other lengths are padded. A stage is one parallel step with no
          cost attached: on a GPU it would be a barrier interval in shared memory, or shuffles between registers.
        </p>
      </Simplification>
      <Network />
      <p>
        The network sorts the ties preset correctly and reorders tied tokens. The same comparators fire however you shuffle
        the input; only which of them swap changes. With the index comparison ticked, every tie group comes out in increasing
        token order.
      </p>
      <Ask q={questionsS149['ch14/14.9/q-comparators']} label="Work it out" />
      <p>
        This is where the networks show up in the code you will read. llama.cpp's CUDA argsort sorts a row with a bitonic
        network in one block's shared memory when the row has at most 1,024 columns and its padded length fits; longer rows
        go to CUB. PyTorch on CUDA sorts slices of 32 or fewer with a bitonic network. Both are the book's "small sequences".
      </p>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md, D1 and D5)">
        <p>
          <code>ggml/src/ggml-cuda/argsort.cu</code>: "Bitonic sort of one row per block". PyTorch,{' '}
          <code>Sort.cpp</code>: <code>bitonicSortKVInPlace</code> "For very small unstable sorts (n &lt;= 32)". On HIP,
          llama.cpp has no CUB path, so long rows take another route.
        </p>
      </Today>
      <Ask q={questionsS149['ch14/14.9/q-network-ties']} />

      <h2>Sorting by partitioning: sample sort</h2>
      <p>
        Comparison sorts without a fixed pattern come in two broad kinds. Some cut the input into tiles, sort each tile and
        do most of their work combining the tiles: merge sort, section 14.3. Others do most of their work partitioning, so
        that combining is trivial. Sample sort is the typical one: choose p − 1 keys, sort them, and use them as splitters to
        cut the input into p buckets, each holding only keys between two neighbouring splitters. Sort every bucket on its
        own, then concatenate. It generalises quicksort's split in two to a split in p.
      </p>
      <Source section="14.9" pages="346" />
      <p>
        The splitters are usually chosen from a larger sample than p − 1 keys, because modest oversampling gives buckets of
        nearly equal size with high probability. The book recommends sample sort for very large inputs spread over several
        physical memories, including the memories of several GPUs in one node; we have not checked that recommendation.
      </p>
      <Ask q={questionsS149['ch14/14.9/q-oversample']} />

      <h2>LSD and MSD radix sort</h2>
      <p>
        Radix sort has the same split between bottom-up and top-down. The chapter's radix sort is least-significant-digit
        first (LSD): every pass reorders the whole array, so every pass shuffles data globally. A most-significant-digit
        (MSD) radix sort partitions by the top digit first, then sorts each bucket on its own by the next digit, and so on;
        each step works on a smaller, more local region. Like sample sort, it is often the better choice for very large
        inputs.
      </p>
      <Source section="14.9" pages="346" />
      <Ask q={questionsS149['ch14/14.9/q-msd']} />

      <h2>Top-k is a selection, not a sort</h2>
      <p>
        A sampler's top-k needs the k largest probabilities, not the order of the other hundred thousand. MSD thinking
        applied to selection does exactly that: count a histogram of the top digit, walk the bins from the top until k keys
        are covered, keep every bucket above that bin, and refine only the bin where the k-th largest key lies. That is what
        llama.cpp's <code>top_k_radix_cuda</code> does (8-bit digits, per-block histograms in shared memory, a select that
        walks bins from the top), and what PyTorch's multi-block top-k, <code>mbtopk</code>, does. Selection is the
        subject of <a href={href('x1')}>module X1</a>.
      </p>
      <Beyond>
        <p>
          Not in the book, which describes no selection algorithm. Sources: notes/today/ch14-facts.md D2, D5.
        </p>
      </Beyond>
      <p>
        Top-p is different: the common formulation sorts the probabilities, takes a running sum (chapter 11's scan) and cuts
        where the sum reaches p. Hugging Face transformers, mlx-lm and llama.cpp's backend sampler all do it this way. It is
        not the only way: vLLM's default Triton path finds the cut with a pivot search and no sort at all.
      </p>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md, A7, B4, C1, D4, D7)">
        <p>
          Libraries: CUB's <code>DeviceTopK</code> currently accepts only the unordered, non-deterministic,
          unspecified-ties configuration; <code>DeviceBatchedTopK</code> needs CUDA 13.5 and caps a segment at 2^21 items.
          rocPRIM has <code>topk</code> and <code>segmented_topk</code>, with no hipCUB wrapper. Apple documents MPSGraph{' '}
          <code>topK</code> (macOS 14) and <code>MPSMatrixFindTopK</code> for k up to 16. MLX's <code>topk</code> runs a
          full sort on Metal.
        </p>
      </Today>

      <h2>What to call instead of writing it</h2>
      <p>
        The book's closing advice is to use a library such as Thrust or CUB rather than write a sorting kernel from scratch,
        and to treat sorting as a case study. For the jobs a sampler has, these are the calls today. The names come from
        each vendor's documentation and headers; nothing in this course compiles CUB or hipCUB, so none of it is verified on
        hardware here.
      </p>
      <Source section="14.10" pages="347" />
      <Today checked="2026-11-20 (notes/today/ch14-facts.md, A2 to A6, B1 to B5, C1)">
        <p>
          On Metal there is nothing to call for a stable row sort over buffers, which is one reason the chapter's labs write
          their own radix sort for every backend. The <a href={href('ch14', 'labs')}>labs page</a> lists them.
        </p>
      </Today>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="libraries">
        <thead>
          <tr><th>Job</th><th>CUDA</th><th>HIP (ROCm)</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>Sort (key, token) pairs, largest first, stable</td>
            <td><code>cub::DeviceRadixSort::SortPairsDescending</code>: documented stable, with a documented float-key transform</td>
            <td><code>hipcub::DeviceRadixSort::SortPairsDescending</code>, over <code>rocprim::radix_sort_pairs_desc</code>: documented stable</td>
            <td>No kernel-level library sort. MPSGraph <code>argSort(_:axis:descending:name:)</code> (macOS 13); stability not stated</td>
          </tr>
          <tr>
            <td>Every row of a batch, largest first, stable</td>
            <td><code>cub::DeviceSegmentedRadixSort::SortPairsDescending</code> or <code>cub::DeviceSegmentedSort::StableSortPairsDescending</code>; never <code>DeviceSegmentedSort::SortPairsDescending</code>, documented as not stable</td>
            <td>The same names in hipCUB</td>
            <td>MPSGraph <code>sort</code> or <code>argSort</code> along an axis; stability, tie order and NaN order not stated</td>
          </tr>
          <tr>
            <td>High-level sort</td>
            <td><code>thrust::sort</code>, <code>thrust::stable_sort_by_key</code>: arithmetic keys with less or greater go to CUB's radix sort; a custom comparator lambda goes to merge sort</td>
            <td>rocThrust: the same split, onto rocPRIM</td>
            <td>None</td>
          </tr>
          <tr>
            <td>Top-k without a full sort</td>
            <td><code>cub::DeviceTopK</code> (unordered, non-deterministic only, so far); <code>cub::DeviceBatchedTopK</code> from CUDA 13.5</td>
            <td><code>rocprim::topk</code>, <code>rocprim::segmented_topk</code>; no hipCUB wrapper</td>
            <td>MPSGraph <code>topK</code> (macOS 14); <code>MPSMatrixFindTopK</code>, k at most 16</td>
          </tr>
          <tr>
            <td>Sort in the kernel language itself</td>
            <td>None; <code>cub::BlockRadixSort</code> inside a kernel</td>
            <td><code>hipcub::BlockRadixSort</code> (its digit-width parameter is ignored)</td>
            <td>None: MSL has no sort, and Apple's sample code has no sorting sample</td>
          </tr>
        </tbody>
      </table>
    </>
  );
}
