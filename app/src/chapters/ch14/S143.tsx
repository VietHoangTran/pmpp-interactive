import { useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  BLOCK_CHOICES,
  N_CHOICES,
  PANEL_KEYS,
  PANEL_SEG,
  blocksOf,
  log2i,
  mergePlan,
  mergeSortStages,
  type StageRow,
  type TieRule,
} from '../../sim/mergestages';

const fmt = (v: number) => v.toLocaleString('en-US');
/** A key with a letter for its place in the input: a came before b. */
const K = (k: number, t?: string) => <>{k}{t ? <sub>{t}</sub> : null}</>;
const keys = (ks: [number, string?][]) => ks.map(([k, t], j) => <span key={j}>{j ? ', ' : ''}{K(k, t)}</span>);

// q-morework: the last stage of a sort of 2^20 keys, with the explorer's steps per search.
const MW_N = 2 ** 20;
const MW_PER_THREAD = 8;
const MW_STEPS = mergePlan({ n: MW_N, segLen: 1024, blocks: 64 }).rows.at(-1)!.stepsPerSearch;
const MW_SEARCH_STEPS = (MW_N / MW_PER_THREAD) * MW_STEPS;
const pow = (n: number) => `2^${log2i(n)} = ${fmt(n)}`;
const range = (bs: number[]) => (bs.length === 1 ? `block ${bs[0]}` : `blocks ${bs[0]}–${bs[bs.length - 1]}`);

/** How the grid is shared in one stage, in words. */
function share(r: StageRow): string {
  const unit = r.stage === 0 ? 'segment' : 'merge';
  return r.blocksPerMerge > 0
    ? `${r.blocksPerMerge} block${r.blocksPerMerge > 1 ? 's' : ''} per ${unit}`
    : `${r.mergesPerBlock} ${unit}s per block`;
}

const TW = 720;
const MAX_ROWS_DRAWN = 64; // segments drawn one by one up to this many

function Tree({ rows, stage, n }: { rows: StageRow[]; stage: number; n: number }) {
  const rh = rows.length > 12 ? 22 : 30;
  const x0 = 70;
  const w = TW - x0 - 10;
  const cur = rows[stage];
  const merges = Math.max(cur.segments, 1);
  return (
    <svg viewBox={`0 0 ${TW} ${rows.length * rh + 54}`} role="img" aria-label="Merge tree" style={{ width: '100%', maxWidth: TW, minWidth: 560, display: 'block' }}
      data-stage={stage} data-segments={cur.segments}>
      {rows.map((r, s) => {
        const y = 8 + s * rh;
        const on = s === stage;
        const many = r.segments > MAX_ROWS_DRAWN;
        return (
          <g key={s} opacity={s > stage ? 0.35 : 1}>
            <text x={x0 - 8} y={y + 16} fontSize="11.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
              {s === 0 ? 'sorted' : `stage ${s}`}
            </text>
            {many ? (
              <g>
                <rect x={x0} y={y} width={w} height={rh - 8} fill="var(--global-wash)" stroke="var(--global)" strokeWidth={on ? 2.2 : 1} />
                <text x={x0 + w / 2} y={y + 15} fontSize="11.5" textAnchor="middle">{fmt(r.segments)} segments of {fmt(r.elementsPerMerge)}</text>
              </g>
            ) : (
              Array.from({ length: r.segments }, (_, k) => (
                <rect key={k} x={x0 + (k * w) / r.segments + 1} y={y} width={w / r.segments - 2} height={rh - 8} rx="2"
                  fill="var(--global-wash)" stroke="var(--global)" strokeWidth={on ? 2.2 : 1} />
              ))
            )}
          </g>
        );
      })}
      {/* who works on what, under the current stage */}
      <g data-assign="">
        {merges <= 16
          ? Array.from({ length: merges }, (_, m) => (
              <text key={m} x={x0 + ((m + 0.5) * w) / merges} y={rows.length * rh + 26} fontSize="11" textAnchor="middle" data-merge={m}>
                {range(blocksOf(cur, m))}
              </text>
            ))
          : (
            <text x={x0 + w / 2} y={rows.length * rh + 26} fontSize="11.5" textAnchor="middle">
              {fmt(merges)} {stage === 0 ? 'segments' : 'merges'}: {share(cur)}
            </text>
          )}
        <text x={x0} y={rows.length * rh + 46} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
          {n.toLocaleString('en-US')} keys; the stage drawn in bold, {stage === 0 ? 'each segment sorted on its own' : `each box is the output of one merge of two ${fmt(cur.elementsPerMerge / 2)}-key inputs`}
        </text>
      </g>
    </svg>
  );
}

function MergeTree() {
  const [n, setN] = useState(2 ** 20);
  const [seg, setSeg] = useState(2 ** 17);
  const [blocks, setBlocks] = useState(8);
  const [stage, setStage] = useState(0);
  const segOptions = Array.from({ length: log2i(n) - 3 }, (_, k) => 2 ** (k + 3)).filter((s) => s < n);
  const segLen = Math.min(seg, n / 2);
  const plan = mergePlan({ n, segLen, blocks });
  const st = Math.min(stage, plan.stages);
  return (
    <Bench
      title="Merge tree"
      task="Start at the book's eight blocks, move the stage slider, then give the grid fewer blocks than segments."
      foot={
        <>
          A search is a co-rank search that finds where one block's share of a merge's output starts in the two inputs:
          a merge shared by b blocks needs b − 1 of them. The keys live in global memory; each stage reads and writes all of
          them once.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Keys
          <select value={n} onChange={(e) => { setN(+e.target.value); setStage(0); }}>
            {N_CHOICES.map((v) => <option key={v} value={v}>{pow(v)}</option>)}
          </select>
        </label>
        <label className="field">
          Initial segment length
          <select value={segLen} onChange={(e) => { setSeg(+e.target.value); setStage(0); }}>
            {segOptions.map((v) => <option key={v} value={v}>{pow(v)}</option>)}
          </select>
        </label>
        <label className="field">
          Blocks in the grid
          <select value={blocks} onChange={(e) => setBlocks(+e.target.value)}>
            {BLOCK_CHOICES.map((v) => <option key={v} value={v}>{fmt(v)}</option>)}
          </select>
        </label>
        <button onClick={() => { setN(2 ** 20); setSeg(2 ** 17); setBlocks(8); setStage(0); }}>The book's eight blocks</button>
      </div>
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <label className="field" style={{ flex: '1 1 16rem' }}>
          Stage: {st === 0 ? 'segments sorted' : `merge stage ${st} of ${plan.stages}`}
          <input type="range" min={0} max={plan.stages} value={st} aria-label="stage" onChange={(e) => setStage(+e.target.value)} />
        </label>
        <span className="ui" data-summary="">
          {fmt(plan.segments)} segments, so {plan.stages} merge stage{plan.stages === 1 ? '' : 's'}
        </span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <Tree rows={plan.rows} stage={st} n={n} />
      </div>
      <table className="threads" aria-label="Stages" style={{ marginTop: '0.8rem' }}>
        <thead>
          <tr><th>stage</th><th>segments after</th><th>merges</th><th>how the grid is shared</th><th>keys per merge</th><th>split searches</th><th>steps per search, about</th></tr>
        </thead>
        <tbody>
          {plan.rows.map((r) => (
            <tr key={r.stage} className={r.stage === st ? 'cur' : ''} data-stage-row={r.stage} data-share={share(r)} data-searches={r.splitSearches}>
              <td>{r.stage === 0 ? 'sort' : r.stage}</td>
              <td>{fmt(r.segments)}</td>
              <td>{r.stage === 0 ? '' : fmt(r.merges)}</td>
              <td>{share(r)}</td>
              <td>{r.stage === 0 ? `${fmt(r.elementsPerMerge)} per segment` : fmt(r.elementsPerMerge)}</td>
              <td>{r.stage === 0 ? '' : fmt(r.splitSearches)}</td>
              <td>{r.stage === 0 ? '' : r.stepsPerSearch}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Bench>
  );
}

function SixteenKeys() {
  const [stage, setStage] = useState(0);
  const [tie, setTie] = useState<TieRule>('left');
  const snaps = mergeSortStages(PANEL_KEYS, PANEL_SEG, tie);
  const s = snaps[Math.min(stage, snaps.length - 1)];
  const final = stage === snaps.length - 1;
  // Tie groups by key, in output order, once the sort is complete.
  const groups = new Map<number, number[]>();
  for (const it of s.items) groups.set(it.key, [...(groups.get(it.key) ?? []), it.id]);
  const tied = [...groups].filter(([, ids]) => ids.length > 1);
  const okKey = (k: number) => {
    const ids = groups.get(k)!;
    return ids.every((v, j) => j === 0 || ids[j - 1] < v);
  };
  return (
    <Bench
      title="Sixteen keys, merged for real"
      task="Step through the stages, then switch the tie rule and look at the repeated keys."
      foot={<>Small numbers under each key are its position in the input, so you can see whether equal keys kept their order. Eight segments of two, as the book's eight blocks would have.</>}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Stage">
          {snaps.map((x) => (
            <button key={x.stage} aria-pressed={x.stage === stage} onClick={() => setStage(x.stage)}>
              {x.stage === 0 ? 'segments sorted' : `merge ${x.stage}`}
            </button>
          ))}
        </span>
        <span className="seg" role="group" aria-label="On equal keys">
          <button aria-pressed={tie === 'left'} onClick={() => setTie('left')}>ties: take the left run first</button>
          <button aria-pressed={tie === 'right'} onClick={() => setTie('right')}>ties: take the right run first</button>
        </span>
      </div>
      <div className="mx-label">Input</div>
      <div className="mono muted" style={{ marginBottom: '0.6rem' }}>{PANEL_KEYS.join(' ')}</div>
      <div className="mx-label">After {s.stage === 0 ? 'sorting each segment' : `merge stage ${s.stage}`}: runs of {s.segLen}</div>
      <div className="mx global" data-row="panel" data-values={s.items.map((x) => `${x.key}:${x.id}`).join(' ')}
        style={{ display: 'flex', flexWrap: 'wrap', rowGap: 4, columnGap: 2 }}>
        {s.items.map((it, k) => (
          <span key={it.id} className={final && groups.get(it.key)!.length > 1 ? (okKey(it.key) ? 'good' : 'bad') : ''}
            style={{ marginLeft: k > 0 && k % s.segLen === 0 ? '0.55rem' : 0, minWidth: '1.9rem', height: 'auto', padding: '0.15rem 0', lineHeight: 1.2 }}>
            <b>{it.key}</b>
            <small className="muted">{it.id}</small>
          </span>
        ))}
      </div>
      {final ? (
        <p style={{ margin: '0.8rem 0 0' }} data-ties={tied.every(([k]) => okKey(k)) ? 'stable' : 'reordered'}>
          {tied.every(([k]) => okKey(k)) ? (
            <span className="chip ok">every repeated key in input order: stable</span>
          ) : (
            <span className="chip fail">repeated keys reordered: {tied.filter(([k]) => !okKey(k)).map(([k]) => k).join(', ')}</span>
          )}
        </p>
      ) : (
        <p className="muted" style={{ margin: '0.8rem 0 0' }}>Step to the last merge to check the repeated keys.</p>
      )}
    </Bench>
  );
}

export const questionsS143: Record<string, Question> = {
  'ch14/14.3/q-blocks': {
    kind: 'numeric',
    id: 'ch14/14.3/q-blocks',
    prompt: (
      <>
        A grid of 64 blocks sorts 64 segments, one per block, and then merges. In merge stage 4 (counting from 1), how many
        blocks work on each merge?
      </>
    ),
    answer: 16,
    explain: (
      <p>
        Each stage halves the number of segments, so stage s has 64 / 2^s merges: 32, 16, 8, 4. Sixty-four blocks shared
        among 4 merges is 16 per merge, which is 2^s: blocks per merge double every stage.
      </p>
    ),
    source: 'ch. 14 §14.3, pp. 333–334 (the book\'s eight-block example)',
  },
  'ch14/14.3/q-stages': {
    kind: 'numeric',
    id: 'ch14/14.3/q-stages',
    prompt: <>2^20 keys are first sorted in segments of 1,024. How many merge stages follow?</>,
    answer: 10,
    explain: <p>2^20 / 2^10 = 1,024 segments. Each stage halves the count: 512, 256, ..., 1. That is log2(1,024) = 10 stages.</p>,
    source: 'ch. 14 §14.3, p. 334',
  },
  'ch14/14.3/q-morework': {
    kind: 'numeric',
    id: 'ch14/14.3/q-morework',
    prompt: (
      <>
        The last stage of a parallel merge sort merges two runs of 2^19 keys into {fmt(MW_N)} outputs. Each thread writes{' '}
        {MW_PER_THREAD} consecutive outputs, and first finds where they start in the two runs with one co-rank search of
        about {MW_STEPS} steps. How many search steps does the stage make in all?
      </>
    ),
    answer: MW_SEARCH_STEPS,
    unit: 'steps',
    explain: (
      <p>
        {fmt(MW_N)} / {MW_PER_THREAD} = {fmt(MW_N / MW_PER_THREAD)} threads, one search each: {fmt(MW_N / MW_PER_THREAD)} ×{' '}
        {MW_STEPS} = {fmt(MW_SEARCH_STEPS)} steps, {MW_SEARCH_STEPS / MW_N} per output, on top of the one move per output a
        sequential merge makes. Per output that is log2 N / {MW_PER_THREAD}, which grows with N: at 2^30 keys it would be
        30 / 8 = 3.75. So a stage costs about N + (N / {MW_PER_THREAD}) log2 N, and over the log N stages the search term
        is the extra factor: O(N log² N) work against O(N log N) for the sequential sort.
      </p>
    ),
    source: 'our reading of ch. 14 §14.3, p. 334 (notes/ch14.md, "Unclear in the PDF" item 9); steps per search: ch. 13 §13.4 and src/sim/mergestages.ts',
  },
  'ch14/14.3/q-stable': {
    kind: 'choice',
    id: 'ch14/14.3/q-stable',
    prompt: (
      <>
        A merge takes from the left run when <code>A[i] &lt; B[j]</code> and from the right run otherwise. It merges the
        left run {keys([[2, 'a'], [5, 'a'], [7]])} with the right run {keys([[2, 'b'], [5, 'b'], [9]])}, where a marks a
        key that came earlier in the input than its equal b. What does it output?
      </>
    ),
    options: [
      { label: keys([[2, 'b'], [2, 'a'], [5, 'b'], [5, 'a'], [7], [9]]), correct: true },
      {
        label: keys([[2, 'a'], [2, 'b'], [5, 'a'], [5, 'b'], [7], [9]]),
        why: 'That is what A[i] <= B[j] gives: on a tie it takes the left run. With <, a tie is not "less", so the right run goes first.',
      },
      {
        label: keys([[2, 'a'], [2, 'b'], [5, 'b'], [5, 'a'], [7], [9]]),
        why: 'Both ties meet the same test, so they come out the same way.',
      },
    ],
    explain: (
      <p>
        At each tie <code>A[i] &lt; B[j]</code> is false, so the merge takes the right run's key, which came later in the
        input: every b overtakes its a. The output is sorted and not stable. Chapter 13's merge writes{' '}
        <code>A[i] &lt;= B[j]</code> for exactly this reason; in the panel above, "ties: take the right run first" is this
        merge. Taking the left run on ties keeps equal keys in input order at every merge, and so through the whole tree,
        provided stage 0's segment sort is stable too.
      </p>
    ),
    source: 'ch. 13 §13.1, pp. 304–305; src/sim/mergestages.ts (mergeRuns); notes/today/ch14-facts.md D4',
  },
};

export function S143() {
  return (
    <>
      <p>
        Odd-even sort is simple and does far too much work. Merge sort is a comparison sort that parallelises well: divide
        the list into segments, sort each segment with any sort you like, then merge sorted segments in pairs, stage after
        stage, until one segment is left.
      </p>
      <Source section="14.3" pages="333" figs="Fig. 14.3" />
      <p>
        In the book's figure a grid of eight blocks sorts eight segments, one each. The first merge stage has four merges,
        so two blocks share each merge; the next has two merges, four blocks each; the last has one merge and all eight
        blocks. Early stages have many small merges, so blocks can simply work on different merges. Late stages have a few
        large merges, so several blocks must share one, and that needs{' '}
        <a href={href('ch13', '13.3')}>chapter 13's co-rank search</a>: each block finds where its share of the output
        starts in the two inputs, then merges its share on its own.
      </p>
      <Source section="14.3" pages="333–334" />
      <p>
        Set up the book's eight blocks in the explorer and slide through the stages. Then keep the keys and give the grid
        fewer blocks than there are segments, and watch where in the tree blocks start to share merges.
      </p>
      <Simplification>
        <p>
          The explorer counts merges, blocks and the searches that split merges between blocks. It has no timing, treats
          every block as equally fast, and does not count the further searches chapter 13's kernels make inside a block.
        </p>
      </Simplification>
      <MergeTree />
      <p>
        The parallelism moves from across merges to within a merge: the number of merges halves each stage while the
        blocks per merge double, and the number of stages is log2 of the number of segments.
      </p>
      <Ask q={questionsS143['ch14/14.3/q-blocks']} label="Work it out" />
      <Ask q={questionsS143['ch14/14.3/q-stages']} label="Work it out" />

      <h2>Why it does more work than the sequential sort</h2>
      <p>
        There are O(log N) merge stages. The book gives each stage O(log N) time and O(N log N) work with enough resources,
        "as seen in Chapter 13", so the whole sort takes O(log² N) time and O(N log² N) work. Sequential merge sort does
        O(N log N) work; the extra log factor comes from the co-rank searches a parallel merge makes and a sequential merge
        does not.
      </p>
      <Source section="14.3" pages="334" />
      <p>
        The PDF is loose here. Chapter 13 states O(m + n) for the sequential merge and about log2 N steps for one co-rank
        search, but never a bound for a whole parallel merge. Our reading of the book's figure: if each thread produces a
        fixed number of outputs and searches once for where they start, a stage of N keys makes on the order of N searches
        of log N steps each, which is the O(N log N) per stage. Cole's algorithm reaches O(log N) time and O(N log N) work;
        the book names it and does not cover it, and neither do we.
      </p>
      <Beyond title="Our reading">
        <p>
          The per-stage derivation is ours (notes/ch14.md, "Unclear in the PDF" item 9). The explorer's steps per search
          are ceil(log2(input length + 1)), chapter 13's "about log2 N"; the course's model of that search
          (src/sim/corank.ts) stays within one step more.
        </p>
      </Beyond>
      <Ask q={questionsS143['ch14/14.3/q-morework']} label="Work it out" />

      <h2>Stability</h2>
      <p>
        A merge sort is stable only if every merge is. When the two runs hold equal keys, the merge must take from the left
        run first, because the left run holds elements that came earlier in the input. Chapter 13's merge does exactly that
        with its <code>A[i] &lt;= B[j]</code> test.
      </p>
      <Source section="13.1" pages="304–305" />
      <Simplification>
        <p>
          The panel runs each merge as a sequential merge, one at a time. A parallel merge that splits by co-rank gives
          the same output for the same tie rule; the panel shows the result, not the schedule.
        </p>
      </Simplification>
      <SixteenKeys />
      <Ask q={questionsS143['ch14/14.3/q-stable']} />

      <h2>Exercise 4, in outline</h2>
      <p>
        The book leaves the implementation to you. One way to lay it out: one block sorts each segment in shared memory;
        then log2(segments) merge stages run, each reading from one buffer and writing to the other, swapping the two between
        stages. In each stage every merge gets grid / merges blocks, which use the co-rank search to split the merge; when
        there are fewer blocks than merges, each block loops over several. A segment left without a partner (when the count
        is not a power of two) is copied across unchanged. This is a sketch, not one of this chapter's labs.
      </p>
      <Source section="14.11" pages="347" />

      <h2>Where you will meet it</h2>
      <p>
        MLX's GPU sort is a merge sort built this way. Each thread first sorts the few values in its registers with the
        odd-even loop of <a href={href('ch14', '14.2')}>section 14.2</a>; a block then merges those runs, finding each
        thread's split with a binary search along the diagonal, which is chapter 13's co-rank; longer rows go through
        several blocks, ping-ponging between buffers. Its merge takes from the right run only when that element is
        strictly smaller, so equal keys stay in input order and the sort is stable, as MLX documents.
      </p>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md, D2 and D4)">
        <p>
          On Metal, MLX's <code>topk</code>, <code>partition</code> and <code>argpartition</code> run this full merge sort
          today ("We direct arg partition to sort for now"). llama.cpp's Metal argsort merges sorted runs across
          threadgroups with the same co-rank partition, preferring the left run on equal values.
        </p>
      </Today>
    </>
  );
}
