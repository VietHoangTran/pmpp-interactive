import { useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { tieGroups, type Item } from '../../sim/sortnet';
import {
  BATCH,
  BATCH_PER_ROW,
  BATCH_ROWS,
  CANDIDATE_PRESETS,
  cascade,
  rowChecks,
  sortCandidates,
  type BatchItem,
  type PassKey,
  type SortKind,
} from '../../sim/sortstable';

/** A row of cells that wraps on a narrow screen. */
const WRAP_ROW = { display: 'flex', flexWrap: 'wrap', gap: 2, marginBottom: '0.8rem' } as const;
/** 0.15, 0.20, 0.125: two decimals unless a third is needed. */
const fmtP = (p: number) => p.toFixed(3).replace(/0$/, '');

/** One candidate: probability over token id; marked when it sits in a tie group. */
function Cand({ it, mark }: { it: Item; mark?: 'good' | 'bad' }) {
  return (
    <span className={mark ?? ''} style={{ height: 'auto', minWidth: '3.3rem', padding: '0.2rem 0.25rem', lineHeight: 1.25 }}>
      <b>{fmtP(it.key)}</b>
      <small className="muted">t{it.id}</small>
    </span>
  );
}

function TiesExplorer() {
  const [preset, setPreset] = useState(0);
  const [kind, setKind] = useState<SortKind>('network');
  const [byIndex, setByIndex] = useState(false);
  const input = CANDIDATE_PRESETS[preset].items;
  const out = sortCandidates(input, kind, byIndex);
  const groups = tieGroups(out);
  const markOf = (it: Item) => {
    const g = groups.find((x) => x.key === it.key);
    return g ? (g.increasing ? 'good' : 'bad') : undefined;
  };
  return (
    <Bench
      title="Ties in a candidate list"
      task="Find the setting that reorders tied tokens, then the one change that fixes it without changing the sort."
      foot={
        <>
          Eight candidates, probability over token id, arriving in increasing token order as a row of a softmax does. The
          output is largest first. Tied cells are marked: success when the tied tokens come out in increasing token
          order, failure when they do not.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Candidates">
          {CANDIDATE_PRESETS.map((p, k) => (
            <button key={p.id} aria-pressed={k === preset} onClick={() => setPreset(k)}>{p.label}</button>
          ))}
        </span>
        <span className="seg" role="group" aria-label="Sort">
          <button aria-pressed={kind === 'stable'} onClick={() => setKind('stable')}>stable sort</button>
          <button aria-pressed={kind === 'network'} onClick={() => setKind('network')}>bitonic network (14.9)</button>
        </span>
        <label className="check">
          <input type="checkbox" checked={byIndex} onChange={(e) => setByIndex(e.target.checked)} /> break ties by token
          index: sort (value, index) pairs
        </label>
      </div>
      <div className="mx-label">Input</div>
      <div className="mx" data-row="input" style={WRAP_ROW}>
        {input.map((it) => <Cand key={it.id} it={it} />)}
      </div>
      <div className="mx-label">Sorted, largest first</div>
      <div className="mx" data-row="output" style={{ ...WRAP_ROW, marginBottom: '0.9rem' }}>
        {out.map((it) => <Cand key={it.id} it={it} mark={markOf(it)} />)}
      </div>
      {groups.length ? (
        <table className="threads" aria-label="Tie groups">
          <thead>
            <tr><th>value</th><th>tokens, in output order</th><th>in token order?</th></tr>
          </thead>
          <tbody>
            {groups.map((g) => (
              <tr key={g.key} data-group={fmtP(g.key)} data-ok={g.increasing ? 'yes' : 'no'}>
                <td>{fmtP(g.key)}</td>
                <td>{g.ids.map((i) => 't' + i).join(', ')}</td>
                <td>{g.increasing ? <span className="chip ok">yes</span> : <span className="chip fail">reordered</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="muted" data-group="none" style={{ margin: 0 }}>No two values are equal, so there is no tie order to keep: every correct sort gives this output.</p>
      )}
    </Bench>
  );
}

const ROW_KEY_LABEL: Record<PassKey, string> = { value: 'by value, largest first', row: 'by row id' };

function BatchCells({ items, label, name }: { items: BatchItem[]; label: string; name: string }) {
  const cell = (t: BatchItem) => (
    <span key={`${t.row}-${t.token}`} style={{ height: 'auto', minWidth: '3.1rem', padding: '0.2rem 0.25rem', lineHeight: 1.25 }}>
      <b>{fmtP(t.p)}</b>
      <small className="muted">r{t.row} t{t.token}</small>
    </span>
  );
  // Positions 0 to 5 and 6 to 11: the blocks row 0 and row 1 should end up in.
  return (
    <div style={{ marginBottom: '0.8rem' }}>
      <div className="mx-label">{label}</div>
      <div data-row={name} style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem 0.9rem' }}>
        {[0, 1].map((half) => (
          <div key={half} className="mx" style={{ ...WRAP_ROW, marginBottom: 0 }}>
            {items.slice(half * BATCH_PER_ROW, (half + 1) * BATCH_PER_ROW).map(cell)}
          </div>
        ))}
      </div>
    </div>
  );
}

type Guess = '' | 'ok' | 'not';

function TwoPasses() {
  const [first, setFirst] = useState<PassKey>('row');
  const [stable2, setStable2] = useState(false);
  const [guess, setGuess] = useState<Guess[]>(['', '']);
  const [applied, setApplied] = useState(false);
  const second: PassKey = first === 'value' ? 'row' : 'value';
  const steps = cascade(BATCH, [{ key: first, stable: true }, { key: second, stable: stable2 }]);
  const checks = rowChecks(steps[2], BATCH_ROWS, BATCH_PER_ROW);
  const reset = () => setApplied(false);
  const ready = guess.every((g) => g !== '');
  return (
    <Bench
      title="Two passes for a batch"
      task="Choose the two passes, predict each row, then apply. Find the only combination that works."
      foot={
        <>
          Two rows of six tokens, flattened row by row as a (rows, vocabulary) tensor is in memory. The goal: each row's
          tokens in their own block of six (rows in order), largest first. Pass 1 always uses the stable sort; the unstable
          one is the bitonic network of section 14.9.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Pass 1
          <select value={first} onChange={(e) => { setFirst(e.target.value as PassKey); reset(); }}>
            <option value="value">{ROW_KEY_LABEL.value}</option>
            <option value="row">{ROW_KEY_LABEL.row}</option>
          </select>
        </label>
        <span className="ui">Pass 2: {ROW_KEY_LABEL[second]}, with</span>
        <span className="seg" role="group" aria-label="Pass 2 sort">
          <button aria-pressed={stable2} onClick={() => { setStable2(true); reset(); }}>a stable sort</button>
          <button aria-pressed={!stable2} onClick={() => { setStable2(false); reset(); }}>an unstable sort</button>
        </span>
      </div>
      <BatchCells items={BATCH} label="Input (row 0 | row 1)" name="input" />
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="ui">Your prediction:</span>
        {[0, 1].map((r) => (
          <label key={r} className="field">
            Row {r}
            <select value={guess[r]} onChange={(e) => { const g = guess.slice(); g[r] = e.target.value as Guess; setGuess(g); reset(); }}>
              <option value="">choose</option>
              <option value="ok">contiguous and sorted</option>
              <option value="not">not both</option>
            </select>
          </label>
        ))}
        <button className="primary" disabled={!ready} onClick={() => setApplied(true)}>Apply both passes</button>
      </div>
      {applied ? (
        <>
          <BatchCells items={steps[1]} label={`After pass 1 (${ROW_KEY_LABEL[first]}, stable)`} name="pass1" />
          <BatchCells items={steps[2]} label={`After pass 2 (${ROW_KEY_LABEL[second]}, ${stable2 ? 'stable' : 'unstable'})`} name="pass2" />
          <table className="threads" aria-label="Row checks">
            <thead>
              <tr><th>row</th><th>contiguous, in row order</th><th>sorted, largest first</th><th>your prediction</th></tr>
            </thead>
            <tbody>
              {checks.map((c) => {
                const right = (guess[c.row] === 'ok') === c.ok;
                return (
                  <tr key={c.row} data-rowcheck={c.row} data-ok={c.ok ? 'yes' : 'no'} data-contiguous={c.contiguous ? 'yes' : 'no'} data-sorted={c.sorted ? 'yes' : 'no'}>
                    <td>row {c.row}</td>
                    <td>{c.contiguous ? <span className="chip ok">yes</span> : <span className="chip fail">no</span>}</td>
                    <td>{c.sorted ? <span className="chip ok">yes</span> : <span className="chip fail">no</span>}</td>
                    <td data-guess={right ? 'right' : 'wrong'}>{right ? 'right' : 'wrong'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      ) : (
        <p className="muted" style={{ margin: 0 }}>Make a prediction for both rows to apply the passes.</p>
      )}
    </Bench>
  );
}

export const questionsS141: Record<string, Question> = {
  'ch14/14.1/q-cascade': {
    kind: 'choice',
    id: 'ch14/14.1/q-cascade',
    prompt: (
      <>
        Take the book's list of (age, income) tuples, [(30,150), (32,80), (22,45), (29,80)]. Sort it by age, ascending.
        Then sort the result by income, nonincreasing, with a stable sort. What comes out?
      </>
    ),
    options: [
      {
        label: '(30,150), (32,80), (29,80), (22,45)',
        why: 'That is the stable sort of the original list by income alone: the 80s in input order. The first pass changed that order.',
      },
      { label: '(30,150), (29,80), (32,80), (22,45)', correct: true },
      {
        label: 'Either order of the two 80s; a stable sort makes no promise here.',
        why: 'A stable sort makes exactly this promise: equal incomes keep the order they had going in, which pass 1 set by age.',
      },
    ],
    explain: (
      <p>
        After pass 1 the order is (22,45), (29,80), (30,150), (32,80). Pass 2 is by income; the two incomes of 80 are equal
        keys, so a stable sort keeps them as pass 1 left them: 29 before 32. The result is sorted by income, and within
        equal income by age. That is the cascade: the secondary key first, the primary key last.
      </p>
    ),
    source: 'ch. 14 §14.1, p. 330',
  },
  'ch14/14.1/q-ties': {
    kind: 'numeric',
    id: 'ch14/14.1/q-ties',
    prompt: (
      <>
        A sampler holds a row of bf16 probabilities, each in [0, 1], over a vocabulary of 151,936 tokens. At least how many
        tokens must have the same value as some token earlier in the row?
      </>
    ),
    answer: 135679,
    explain: (
      <>
        <p>
          bf16 can represent at most 16,257 distinct values in [0, 1]: the bit patterns 0x0000 to 0x3F80, the last being
          1.0 (negative zero adds a pattern but not a value). 151,936 tokens into 16,257 values: at least 151,936 − 16,257 =
          135,679 of them repeat a value already seen. A real row has far fewer distinct values than that, since most
          probabilities are tiny.
        </p>
        <p>
          Ties are therefore guaranteed, not rare, and their order is part of what a sampler produces: after a sort, which
          token sits at a given rank decides which token a given random draw selects.
        </p>
      </>
    ),
    source: 'our arithmetic; vocabulary size from notes/today/ch09-facts.md D7',
  },
  'ch14/14.1/q-unstable-second': {
    kind: 'choice',
    id: 'ch14/14.1/q-unstable-second',
    prompt: (
      <>
        You sort a flattened (rows, vocabulary) batch in two passes: by value over everything, then by row id. The row-id
        pass uses an unstable sort. What comes out?
      </>
    ),
    options: [
      { label: 'Rows contiguous and in row order, but the values inside each row in no promised order.', correct: true },
      {
        label: 'Fully correct: row ids are integers, so the second pass cannot go wrong.',
        why: 'It cannot get the rows wrong. But every token of a row has the same row id, so inside a row all keys tie, and an unstable sort may shuffle them.',
      },
      { label: 'Rows interleaved.', why: 'The last pass sorts by row id; whatever else it does, it groups the rows.' },
    ],
    explain: (
      <p>
        The last pass decides the primary order (rows), and only its stability carries the earlier pass's order (values)
        through. Inside one row every key of the row-id pass is equal: a tie group of the whole row. In the explorer above,
        choose value first and an unstable second pass: both rows come out contiguous and unsorted.
      </p>
    ),
    source: 'ch. 14 §14.1, p. 330; notes/today/ch14-facts.md D5',
  },
  'ch14/14.1/q-bound': {
    kind: 'choice',
    id: 'ch14/14.1/q-bound',
    prompt: <>Why can radix sort do better than the bound of order N log N comparisons?</>,
    options: [
      { label: 'It uses more threads.', why: 'The bound is on comparisons, however many threads make them.' },
      {
        label: 'It works out each key\'s position from its digits and never compares two keys; the bound counts comparisons, so it does not apply. The price: keys must be bit patterns that order correctly.',
        correct: true,
      },
      { label: 'The bound only holds for sequential code.', why: 'It holds for any sort that learns about the order only by comparing pairs.' },
      { label: 'It does not; radix sort is N log N too.', why: 'For a fixed key width, the number of passes is fixed and each pass is linear in N.' },
    ],
    explain: (
      <p>
        A comparison sort learns about the order one comparison at a time, and that is what the bound counts. Radix sort
        never asks "is a less than b?": it reads a digit and computes where the key goes. That works only for keys whose
        bits, read as an unsigned integer, order the way the values do. Unsigned integers do; floats do not, which section
        14.4 onward has to deal with.
      </p>
    ),
    source: 'ch. 14 §14.1, p. 330',
  },
};

export function S141() {
  return (
    <>
      <p>
        A sort has two jobs. The output must be in order, nondecreasing or nonincreasing, and it must be a permutation of
        the input: every element kept, none invented. The second job sounds trivial and is the one a careless parallel
        sort breaks first, as the next section shows.
      </p>
      <Source section="14.1" pages="330" />
      <p>
        The book's interesting case is elements with a key and a value: tuples of (age, income), sorted by income. In a
        sampler the key is a probability and the value is its token id. An argsort is exactly this: a key-value sort whose
        values start as 0, 1, ..., V − 1, so that after sorting the values say which token sits at each rank.
      </p>

      <h2>Stability, and why a sampler needs it</h2>
      <p>
        Sort the book's list [(30,150), (32,80), (22,45), (29,80)] by income, largest first. Both (32,80) and (29,80) have
        the key 80. A <b>stable</b> sort promises that equal keys keep their input order, so (32,80) stays first. An
        unstable sort promises only a correct order of keys; either 80 may come first, and two runs may disagree.
      </p>
      <Source section="14.1" pages="330" />
      <p>
        For a sampler that holds its probabilities in bf16, ties are not an edge case. A large vocabulary has far more
        tokens than bf16 has values in [0, 1], so many tokens must share a value. Their order after the sort decides which
        token sits at each rank, and therefore which token a given random number selects. A stable sort fixes that order
        to increasing token index; an unstable one leaves it to the implementation.
      </p>
      <Beyond>
        <p>
          The sampler reading is ours. The book discusses stability for multi-key sorting and, in section 14.4, for radix
          sort's correctness.
        </p>
      </Beyond>
      <Ask q={questionsS141['ch14/14.1/q-ties']} label="Work it out" />
      <p>
        The explorer sorts eight candidates largest first, with a stable sort or with one particular unstable sort: the
        bitonic network you will meet in section 14.9. Start with ties at two values and the network.
      </p>
      <Simplification>
        <p>
          The unstable sort here is one fixed network. An unstable sort promises nothing, so another one could reorder
          ties differently or, by luck, not at all. Probabilities are short decimals, not bf16 bit patterns.
        </p>
      </Simplification>
      <TiesExplorer />
      <p>
        Three things to see. The network reorders one tie group and leaves the other alone. Ticking the index option
        makes the network agree with the stable sort: comparing (value, index) pairs means no two keys are ever equal, so
        any correct sort produces the same output. And with all eight values equal, the network keeps the input order,
        because none of its comparators ever finds a pair out of order: stability by luck, not by promise.
      </p>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md, D4 and D5)">
        <p>
          PyTorch's <code>torch.sort</code> defaults to <code>stable=False</code>, and on CUDA sorts slices of 32 or fewer
          with an unstable bitonic network. <code>torch.topk</code> documents that tied indices "are not guaranteed to be
          stable". MLX documents <code>mlx.core.sort</code> as stable.
        </p>
      </Today>

      <h2>Sorting by two keys</h2>
      <p>
        Stability is what makes multi-key sorting work. To sort by a primary key and, within it, by a secondary key, sort
        by the secondary key first, then by the primary key with a stable sort. The second pass orders by the primary key
        and, among equal primary keys, keeps the order the first pass made.
      </p>
      <Source section="14.1" pages="330" />
      <Ask q={questionsS141['ch14/14.1/q-cascade']} />
      <p>
        The same cascade can sort every row of a batch at once: flatten the rows into one array and sort it twice, once by
        value and once by row id. In which order, and which pass must be stable? The explorer below gives you two rows;
        predict each one before you apply the passes.
      </p>
      <TwoPasses />
      <p>
        Only one combination works: values first, row id last, and the last pass stable. Rows first fails however stable
        the second pass is, because a pass by value pays no attention to rows.
      </p>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md, D5)">
        <p>
          PyTorch on CUDA sorts the rows of a tensor this way in <code>segmented_sort_pairs_by_full_sort</code>{' '}
          (<code>SortStable.cu</code>): "First we sort by values, globally ... Then we stable sort by segment id". It is
          used for 2 to 127 segments shorter than a million elements. One segment, or very long ones, take a per-segment
          radix sort; 128 segments or more, CUB's segmented radix sort.
        </p>
      </Today>
      <Ask q={questionsS141['ch14/14.1/q-unstable-second']} />

      <h2>Comparison and non-comparison sorts</h2>
      <p>
        A sort that learns about the order only by comparing pairs of keys needs on the order of N log N comparisons for N
        keys; no comparison sort does better. A non-comparison sort can, at a price: it does not work for arbitrary key
        types. Both kinds can be parallelised.
      </p>
      <Source section="14.1" pages="330" />
      <p>
        Radix sort, the chapter's main subject, is the non-comparison kind. It reads each key one digit at a time and
        computes where the key goes, which needs keys whose bits order the way the values do when read as an unsigned
        integer. Unsigned integers qualify. Floats do not: a negative float has its sign bit set, so read as an unsigned
        integer it looks larger than every positive one. Probabilities, which are never negative, happen to order
        correctly; logits do not, and the transform that fixes them comes with radix sort.
      </p>
      <Beyond>
        <p>
          The float example is ours. The transform is documented by CUB for its radix sort (notes/today/ch14-facts.md A3).
        </p>
      </Beyond>
      <Ask q={questionsS141['ch14/14.1/q-bound']} />

      <h2>The chapter's map</h2>
      <p>
        The chapter uses sorting as a case study in parallelisation and optimisation, after Satish, Harris and Garland. It
        covers two comparison sorts, <a href={href('ch14', '14.2')}>odd-even sort</a> and{' '}
        <a href={href('ch14', '14.3')}>merge sort</a>, briefly, and one non-comparison sort,{' '}
        <a href={href('ch14', '14.4')}>radix sort</a>, at length (sections 14.4 to 14.8): a pass is a stable partition
        built from an exclusive scan, and the optimisations are those of chapter 12's stable filter.{' '}
        <a href={href('ch14', '14.9')}>Section 14.9</a> surveys the rest, and is where the library calls you should
        normally use instead are listed.
      </p>
      <Source section="14.1" pages="330–331" />
    </>
  );
}
