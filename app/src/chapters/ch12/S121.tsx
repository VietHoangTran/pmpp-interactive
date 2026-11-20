import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  FIG12_1,
  arrivalOrder,
  crossings,
  inPlaceOverwrites,
  keepFrom,
  keptOf,
  sameKeys,
  stableFilter,
  unstableFilter,
} from '../../sim/filter';

// Section 12.1, Background (pp. 289-290). The explorer re-draws Fig. 12.1 from
// src/sim/filter.ts; the classification is static data. Vendor and library facts:
// notes/today/ch12-facts.md (A7, A8, B5, C5) and ch09-facts.md (A3, B1).

const N = FIG12_1.n;
const CELL = 34;
const PITCH = 38;
const LEFT = 70;
const SVG_W = LEFT + N * PITCH + 8;
const IN_Y = 22;
const OUT_Y = 100;

const isBook = (keep: boolean[]) => keptOf(keep).join(',') === FIG12_1.kept.join(',');

/** One panel of Fig. 12.1: the input row, the output row and the arrows between them. */
function Panel({ name, keep, output, inPlace, onToggle }: {
  name: 'stable' | 'unstable';
  keep: boolean[];
  output: number[];
  inPlace: boolean;
  onToggle: (k: number) => void;
}) {
  const over =inPlace && name === 'stable' ? inPlaceOverwrites(keep) : [];
  const h = OUT_Y + CELL + (over.length ? 46 : 18);
  const cx = (i: number) => LEFT + i * PITCH + CELL / 2;
  return (
    <svg viewBox={`0 0 ${SVG_W} ${h}`} role="img" aria-label={`${name} filter`} data-panel={name}
      data-output={output.map((k) => 'k' + k).join(' ')} data-crossings={crossings(output)}
      style={{ width: '100%', minWidth: 540, display: 'block' }}>
      <text x={0} y={IN_Y + 21} fontSize="12">input</text>
      <text x={0} y={OUT_Y + 21} fontSize="12">output</text>
      {output.map((k, s) => (
        <line key={'a' + k} x1={cx(k)} y1={IN_Y + CELL + 2} x2={cx(s)} y2={OUT_Y - 3}
          stroke="var(--ink-2)" strokeWidth="1.3" opacity="0.8" />
      ))}
      {keep.map((kept, i) => (
        <g key={i} data-key={i} data-kept={kept ? 1 : 0} style={{ cursor: 'pointer' }} onClick={() => onToggle(i)}>
          <rect x={LEFT + i * PITCH} y={IN_Y} width={CELL} height={CELL} rx="3"
            fill={kept ? 'var(--global-wash)' : 'var(--cell)'} stroke={kept ? 'var(--global)' : 'var(--rule)'} />
          <text x={cx(i)} y={IN_Y + 21} fontSize="12" textAnchor="middle" style={{ fill: kept ? 'var(--ink)' : 'var(--ink-3)' }}>k{i}</text>
        </g>
      ))}
      {output.map((k, s) => (
        <g key={'o' + s}>
          <rect x={LEFT + s * PITCH} y={OUT_Y} width={CELL} height={CELL} rx="3" fill="var(--global-wash)" stroke="var(--global)" />
          <text x={cx(s)} y={OUT_Y + 21} fontSize="12" textAnchor="middle">k{k}</text>
        </g>
      ))}
      {over.length ? (
        <>
          <text x={0} y={OUT_Y + CELL + 16} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>overwrites</text>
          <text x={0} y={OUT_Y + CELL + 30} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>read first</text>
        </>
      ) : null}
      {over.map((o) => (
        <g key={'h' + o.slot} data-held={o.held} data-moves={o.heldMovesTo ?? ''} data-kind={o.kind}>
          <text x={cx(o.slot)} y={OUT_Y + CELL + 16} fontSize="11" textAnchor="middle"
            style={{ fill: o.kind === 'own' ? 'var(--ink-3)' : 'var(--fail)' }}>{o.kind === 'own' ? 'itself' : `k${o.held}`}</text>
          {o.kind !== 'own' ? (
            <text x={cx(o.slot)} y={OUT_Y + CELL + 30} fontSize="10" textAnchor="middle" style={{ fill: 'var(--fail)' }}>
              {o.kind === 'moved' ? `to ${o.heldMovesTo}` : 'test'}
            </text>
          ) : null}
        </g>
      ))}
    </svg>
  );
}

function StableOrUnstable() {
  const [keep, setKeep] = useState<boolean[]>(() => keepFrom(N, FIG12_1.kept));
  const [run, setRun] = useState(0);
  const [inPlace, setInPlace] = useState(false);
  const book = isBook(keep);
  // Run 0 on the book's keys is the book's own drawing; every other run is a seeded draw.
  const unstable = useMemo(
    () => unstableFilter(keep, run === 0 && book ? [...FIG12_1.bookUnstable] : arrivalOrder(keep, 1000 + run)),
    [keep, run, book],
  );
  const stable = useMemo(() => stableFilter(keep), [keep]);
  const toggle = (k: number) => {
    setKeep((v) => v.map((x, i) => (i === k ? !x : x)));
    setRun(0);
  };
  const set = (kept: number[]) => {
    setKeep(keepFrom(N, kept));
    setRun(0);
  };
  const writes = inPlace ? inPlaceOverwrites(keep) : [];
  const moved = writes.filter((o) => o.kind === 'moved').length;
  const tested = writes.filter((o) => o.kind === 'tested').length;
  const same = sameKeys(stable, unstable);
  const ordered = unstable.every((k, i) => k === stable[i]);
  return (
    <Bench
      title="Stable or unstable?"
      task="Click keys to keep or drop them, then press Run again a few times. What never changes?"
      foot={
        <>
          Fig. 12.1 drawn again: the shaded keys are kept. The first unstable run on the book's keys is the order the
          book draws; each later run is a new completion order.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button className="primary" onClick={() => setRun((r) => r + 1)}>Run again</button>
        <button onClick={() => set([...FIG12_1.kept])} disabled={book}>The book's keys</button>
        <button onClick={() => set([])}>Drop all</button>
        <button onClick={() => set(Array.from({ length: N }, (_, i) => i))}>Keep all</button>
        <label className="check">
          <input type="checkbox" checked={inPlace} onChange={(e) => setInPlace(e.target.checked)} /> show what an in-place stable filter overwrites
        </label>
      </div>
      <div className="mx-label">Stable: kept keys in input order</div>
      <Panel name="stable" keep={keep} output={stable} inPlace={inPlace} onToggle={toggle} />
      <p className="status-line" style={{ margin: '0.3rem 0 0.9rem' }} data-line="stable">
        {stable.length} kept, {crossings(stable)} crossing arrows.
        {inPlace ? (
          moved + tested ? (
            <>
              {' '}In place, {moved + tested} of the {stable.length} writes land on another key, which must be read first:{' '}
              {moved} kept {moved === 1 ? 'key' : 'keys'} that must be read and moved ("to"), and {tested} dropped{' '}
              {tested === 1 ? 'key' : 'keys'} that must still be read, to be tested ("test").
            </>
          ) : stable.length ? (
            <> In place, every kept key is written over itself: nothing to protect.</>
          ) : null
        ) : null}
      </p>
      <div className="mx-label">Unstable: run {run + 1}</div>
      <Panel name="unstable" keep={keep} output={unstable} inPlace={false} onToggle={toggle} />
      <p className="status-line" style={{ margin: '0.3rem 0 0' }} role="status" data-line="unstable">
        {unstable.length} kept, {crossings(unstable)} crossing arrows.{' '}
        <span className={'chip ' + (same ? 'ok' : 'fail')} data-check={same ? 'same' : 'different'}>
          {same ? 'same keys, same count' : 'different keys'}
        </span>{' '}
        {unstable.length > 1 ? (ordered ? 'This run happens to be in input order.' : 'Order differs from the stable output.') : null}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- does it need order?
type Verdict = 'stable' | 'unstable' | 'none';
const VERDICT_LABEL: Record<Verdict, string> = {
  stable: 'stable needed',
  unstable: 'unstable is enough',
  none: 'no compaction needed',
};
const JOBS: { id: string; job: string; answer: Verdict; why: string }[] = [
  {
    id: 'sort-next',
    job: 'Collect the (token id, logit) pairs above a threshold; the next step sorts them by logit, breaking ties by token id.',
    answer: 'unstable',
    why: 'The sort decides the final order, ties included, and each id travels with its logit, so the order the filter produced is thrown away. Without the tie rule, a stable sort would keep tied logits in whatever order the filter left them, and that order changes from run to run.',
  },
  {
    id: 'dedup',
    job: 'Remove the duplicates from a sorted list of token ids.',
    answer: 'stable',
    why: 'The result must still be sorted. Section 12.8 meets this job again: it is a filter whose test is "differs from the key before me".',
  },
  {
    id: 'gather',
    job: 'Find the indices of the nonzero entries for a later gather that expects them in ascending order.',
    answer: 'stable',
    why: 'The consumer depends on the order, so the filter must keep it.',
  },
  {
    id: 'seeded',
    job: 'Draw a token by walking the compacted candidates until a running sum passes a seeded random number; the same seed must give the same token.',
    answer: 'stable',
    why: 'The walk depends on the array order. Any order that is the same every run would do; a stable filter is the simplest way to get one.',
  },
  {
    id: 'count',
    job: 'Count how many logits pass a threshold.',
    answer: 'none',
    why: 'Nothing has to move. A count is a reduction (chapter 10).',
  },
  {
    id: 'kv-free',
    job: 'Collect the ids of free KV-cache blocks, where any free block is as good as another.',
    answer: 'unstable',
    why: 'No later step reads the order. (Our example, not the book\'s.)',
  },
];

function NeedsOrder() {
  const [picks, setPicks] = useState<Record<string, Verdict>>({});
  const right = JOBS.filter((j) => picks[j.id] === j.answer).length;
  const done = Object.keys(picks).length;
  return (
    <Bench
      title="Does it need order?"
      task="Place each job, then read the reason."
      foot={done ? <span data-score={right}>{right} of {JOBS.length} placed as we would place them ({done} answered).</span> : 'Six jobs from an inference server.'}
    >
      <div style={{ display: 'grid', gap: '0.7rem' }}>
        {JOBS.map((j) => {
          const pick = picks[j.id];
          return (
            <div key={j.id} className="panel" data-job={j.id} data-pick={pick ?? ''} data-right={pick ? (pick === j.answer ? 1 : 0) : ''}>
              <p style={{ margin: '0 0 0.5rem' }}>{j.job}</p>
              <div className="seg" role="group" aria-label={'Place: ' + j.id}>
                {(Object.keys(VERDICT_LABEL) as Verdict[]).map((v) => (
                  <button key={v} aria-pressed={pick === v} onClick={() => setPicks((p) => ({ ...p, [j.id]: v }))}>
                    {VERDICT_LABEL[v]}
                  </button>
                ))}
              </div>
              {pick ? (
                <p style={{ margin: '0.5rem 0 0' }}>
                  <span className={'chip ' + (pick === j.answer ? 'ok' : 'fail')}>{pick === j.answer ? 'yes' : `we would say: ${VERDICT_LABEL[j.answer]}`}</span>{' '}
                  {j.why}
                </p>
              ) : null}
            </div>
          );
        })}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS121: Record<string, Question> = {
  'ch12/12.1/q-reproduce': {
    kind: 'choice',
    id: 'ch12/12.1/q-reproduce',
    prompt: (
      <>
        A sampler compacts the (token id, probability) pairs above a threshold with an unstable filter, then draws a token
        by walking the compacted array until the running sum of probabilities passes a seeded random number u. Same
        logits, same seed, two runs. Which holds?
      </>
    ),
    options: [
      {
        label: 'The same token every time: the seed fixes u.',
        why: 'u is fixed, but the walk visits the candidates in array order. Another order puts a different token where the running sum crosses u.',
      },
      {
        label: 'The distribution is wrong, because the probabilities are out of order.',
        why: 'Each id travels with its probability, and the set of pairs is the same in every run. Walking a shuffled list still picks each token with its own probability.',
      },
      {
        label: 'The distribution of draws is right, but the token chosen for a given seed can differ between runs.',
        correct: true,
      },
    ],
    explain: (
      <p>
        The filter fixes the set of candidates, not their order. Sampling only needs the set, so it stays correct in
        distribution. Reproducing a draw from a seed needs the order too. Either keep the input order (a stable filter)
        or make sure the order is the same every run.
      </p>
    ),
    source: 'our analysis of §12.1, p. 290',
  },
  'ch12/12.1/q-inplace-slot': {
    kind: 'numeric',
    id: 'ch12/12.1/q-inplace-slot',
    prompt: (
      <>
        Keys k0 to k7. Keep k1, k3, k4 and k6, compacted in place, in stable order. Into which position does k6 move?
      </>
    ),
    answer: 3,
    unit: 'position',
    explain: (
      <p>
        k1 goes to 0, k3 to 1, k4 to 2, k6 to 3. Position 3 held k3, which itself moves to position 1, so k3 must be read
        before k6 is written over it. That is the hazard the book names on p. 290; section 12.7 orders the reads and the
        writes so it cannot happen.
      </p>
    ),
    source: '§12.1, p. 290; §12.7, p. 300',
  },
  'ch12/12.1/q-which-place': {
    kind: 'choice',
    id: 'ch12/12.1/q-which-place',
    prompt: (
      <>
        A list fills most of a device's memory, and the job only needs the surviving keys afterwards. Out of place or
        in place, and what does in place cost you?
      </>
    ),
    options: [
      {
        label: 'Out of place: in place is impossible on a GPU, because threads run in no fixed order.',
        why: 'Section 12.7 does it, by ordering the reads before the writes they could clobber.',
      },
      {
        label: 'In place, because there is no room for a second list. The cost: threads must be ordered so that no position is written before its old key has been read.',
        correct: true,
      },
      {
        label: 'In place, and it costs nothing extra.',
        why: 'It adds ordering constraints between threads, and between blocks when a key moves into another block\'s range.',
      },
    ],
    explain: (
      <p>
        Out of place needs room for a whole second list, which is exactly what this job does not have. In place keeps
        one list but adds a dependence: a key may move into a position whose old key has not been read yet.
      </p>
    ),
    source: '§12.1, p. 290; §12.7, p. 300',
  },
};

export function S121() {
  return (
    <>
      <p>
        A filter takes a list, removes the items that fail a test, and closes the gaps, so the survivors sit next to
        each other with no holes between them. The book calls the items of the list <b>keys</b>. Seen as two lists, the
        keys kept and the keys dropped, a filter is a merge run backwards: chapter 13 combines two lists into one, a
        filter splits one list and keeps a part.
      </p>
      <Source section="12.1" pages="289–290" />
      <p>
        The book's example is memory management. After a program frees many objects, its heap is a patchwork of small
        free gaps; together they may hold plenty of space, but none may be big enough for one large allocation. Moving
        the live objects together leaves a single free region at the end. For a large heap, many threads moving objects
        at once cut the time.
      </p>

      <h2>Out of place or in place</h2>
      <p>
        An <b>out-of-place</b> filter writes the survivors to a new list. It is the simpler kind and the usual one in
        search and retrieval, where irrelevant items are dropped to build a list of relevant ones. Its price is room
        for that second list; for the heap, room for a second heap.
      </p>
      <Source section="12.1" pages="290" />
      <p>
        An <b>in-place</b> filter compacts the list where it lies, which saves that room and adds a hazard. A key moves
        to an earlier position, and that position may still hold a key that has not been read yet. With the keys of
        Fig. 12.1 below, k3 lands where k1 lived: write k3 before k1 has been read, and k1 is lost. A dropped key needs
        the same care, because its thread must read it to test it before anything lands on top. Section 12.7 shows
        how to order the reads and the writes; the book builds its in-place filters on the scan of chapter 11.
      </p>
      <Ask q={questionsS121['ch12/12.1/q-which-place']} />
      <Ask q={questionsS121['ch12/12.1/q-inplace-slot']} label="Work it out" />

      <h2>Stable or unstable</h2>
      <p>
        A <b>stable</b> filter keeps the survivors in their input order. An <b>unstable</b> filter keeps the same
        survivors in some order, which is fine whenever nothing downstream looks at the order. This one property shapes
        the rest of the chapter. An unstable filter lets each thread grab any free slot (sections 12.2 to 12.4). A
        stable filter must work out exactly where each key goes, and that takes a scan (sections 12.5 to 12.7).
      </p>
      <Source section="12.1" pages="290" figs="Fig. 12.1" />
      <p>
        The explorer draws Fig. 12.1: sixteen keys, the shaded ones kept, a stable filter above and an unstable one
        below. Click keys to change which are kept, press <i>Run again</i> a few times, and watch the arrows and the
        line under each panel. Then tick the in-place box and find the writes that land on a key nobody has read yet.
      </p>
      <Simplification>
        <p>
          The unstable order is one draw of a seeded shuffle, standing for the order in which the threads' atomic
          operations happen to complete. No vendor fixes that order (ch09-facts A3, B1), and this is not a model of how
          any hardware picks it.
        </p>
      </Simplification>
      <StableOrUnstable />
      <p>
        Two things never change from run to run: which keys come out, and how many. Only their order moves. In the
        stable panel no two arrows ever cross, which is just another way to say the order is kept; in the unstable
        panel they usually do.
      </p>
      <Aside>
        <p>
          The PDF captions both panels of Fig. 12.1 "(a)". The second one, the stable filter, should be (b).
        </p>
      </Aside>

      <h2>Does a job need order?</h2>
      <p>
        Unstable is enough when the next step does not read the order. Stable is needed when the list was sorted and
        must stay sorted, and also when a result has to come out the same in every run. Place six jobs from an
        inference server.
      </p>
      <Source section="12.1" pages="290" />
      <NeedsOrder />
      <p>
        Sampling shows why order can matter even when nothing is sorted. Compact the candidate (token id, probability)
        pairs above a threshold, then pick a token by walking the compacted array until a running sum passes a seeded
        random number. Because each id travels with its probability, the distribution of draws is right in any order.
        But the walk visits candidates in array order, so after an unstable filter the same seed can land on a
        different token in the next run.
      </p>
      <Beyond>
        <p>
          The book does not discuss sampling; this argument is ours. Where PyTorch keeps order for the same reason is
          on the <a href={href('ch12', 'wrap')}>summary page</a>.
        </p>
      </Beyond>
      <Ask q={questionsS121['ch12/12.1/q-reproduce']} />
      <p>
        Why learn unstable filters at all, then? Because when order does not matter they do less work. NVIDIA's own
        write-up of this chapter's warp technique puts it plainly: a stable filter is built on a prefix sum and "is
        more expensive as a result", while "a purely atomic approach is simpler and performs less work". The next
        three sections build that atomic approach and make it fast; the <a href={href('ch12', 'labs')}>labs</a> measure
        it.
      </p>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md A7, A8, B5, C5)">
        <p>
          Every library filter found is stable. Thrust's <code>copy_if</code> "is stable"; CUB's{' '}
          <code>DeviceSelect</code> copies "maintain their original relative ordering", and it has in-place overloads.
          No unstable select was found in either. On AMD, hipCUB forwards to rocPRIM, whose documentation shows the
          order only by example. On Apple GPUs the one library compaction found is MPSGraph's{' '}
          <code>nonZeroIndices</code> (macOS 14 and later, <code>int32</code> indices).
        </p>
      </Today>
    </>
  );
}
