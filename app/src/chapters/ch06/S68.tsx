import { useState, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { href } from '../../lib/router';
import {
  CATEGORIES,
  CATEGORY_AIM,
  CATEGORY_NAME,
  OUTCOME_LABEL,
  ROWS,
  SCENARIOS,
  SYMPTOMS,
  categoryCounts,
  countLine,
  rankRows,
  rowById,
  sortScore,
  type Outcome,
  type Placement,
  type Row,
  type SymptomId,
} from '../../sim/checklist';

export const questionsS68: Record<string, Question> = {
  'ch06/6.8/q-tiling-occ': {
    kind: 'choice',
    id: 'ch06/6.8/q-tiling-occ',
    prompt: (
      <>
        A tiled matmul&rsquo;s occupancy is already limited by its shared-memory use, and the profiler shows it rarely
        waits on global memory. Someone has each block of T × T threads compute a 2T × 2T tile of the output, four
        results per thread, to cut global traffic further. Each block now keeps tiles of M and N twice the size: twice
        the shared memory per block, the same threads, half the global traffic. What do you expect?
      </>
    ),
    options: [
      {
        label: 'Slower: it spends more of the resource that limits the kernel to relieve one that does not.',
        correct: true,
      },
      {
        label: 'Faster: tiling always cuts global traffic, and less traffic is always better.',
        why: 'Less traffic helps only if traffic was the limit. Here it was not, and the extra shared memory removes resident blocks.',
      },
      {
        label: 'No change: the traffic falls but nothing else moves.',
        why: 'Something else does move: twice the shared memory per block at the same thread count leaves room for about half as many blocks per SM, so occupancy falls further.',
      },
    ],
    explain: (
      <>
        <p>
          This is the book&rsquo;s own warning on p. 153, and the reason the checklist has a cost column in this
          course. Fewer resident blocks means fewer warps to hide latency with, and fewer memory requests in flight
          (6.2). Section 6.9 lets you watch it happen in a model.
        </p>
        <p>
          Simply doubling <code>TILE_WIDTH</code> in chapter 5&rsquo;s kernel would not test this. Its block is T × T
          threads with one T × T tile of M and one of N, so a tile twice as wide has four times the shared memory and
          four times the threads. Shared memory per thread stays at two floats, 8 bytes, and an occupancy limited by
          shared memory stays where it was, apart from rounding. What spends more of the limiting resource is more
          shared memory per thread.
        </p>
      </>
    ),
    source: 'ch. 6 §6.9, p. 153; occupancy budgets from ch. 5 §5.6',
  },
  'ch06/6.8/q-sectors': {
    kind: 'choice',
    id: 'ch06/6.8/q-sectors',
    prompt: (
      <>
        On an NVIDIA GPU, Nsight Compute reports 32 sectors per request for a kernel&rsquo;s 4-byte global loads; the
        ideal for 4-byte loads by a full warp is 4. Which row of the checklist, and which of its strategies?
      </>
    ),
    options: [
      {
        label:
          'Coalescing: remap threads to data, change the layout (transpose, structure of arrays), or stage through shared memory (corner turning).',
        correct: true,
      },
      { label: 'Occupancy tuning: more warps.', why: 'More warps put more of the same wasteful requests in flight. Each request still fetches 32 sectors for 128 useful bytes.' },
      { label: 'Padding against bank conflicts.', why: 'Bank conflicts are a shared-memory problem. Sectors per request is about global loads.' },
      { label: 'Loop unrolling.', why: 'Unrolling changes how many instructions run, not which addresses each one touches.' },
    ],
    explain: (
      <p>
        32 sectors for 32 threads means every thread&rsquo;s 4 bytes came from a different 32-byte sector: the
        neighbours in the warp are at least 32 bytes apart, the <code>N[col*Width + k]</code> pattern of 6.1. Seven
        eighths of every sector is fetched and not used. The fix is in the addresses, so it is in the coalescing row.
        The sector unit and the metric are NVIDIA&rsquo;s (facts A2, A11, A17); AMD counts cache lines and Apple
        documents no unit.
      </p>
    ),
    source: 'ch. 6 §6.1 and §6.8, pp. 128–129, 150–151; notes/today/ch06-facts.md A2, A11, A17',
  },
  'ch06/6.8/q-general': {
    kind: 'choice',
    id: 'ch06/6.8/q-general',
    prompt: <>Why does the book file thread coarsening under &ldquo;general&rdquo; instead of one of the three categories?</>,
    options: [
      {
        label:
          'The overhead it removes depends on the kernel: redundant loads in stencil and matmul, private copies in histogram, synchronisation and divergence in reduction and scan, poor coalescing in filter and sorting.',
        correct: true,
      },
      { label: 'It has no benefit to memory.', why: 'In matmul its benefit is entirely about memory: fewer redundant tile loads.' },
      { label: 'It is too new to classify.', why: 'It is used throughout Parts 2 and 3; the reason is that its effect changes from kernel to kernel.' },
    ],
    explain: (
      <p>
        Every other row is filed under its primary intent (p. 149). Coarsening has none of its own: it serialises some
        of the parallel work, and what that buys you is whatever overhead the parallel version was paying. That is
        also why it can buy nothing (pitfall 1 on p. 142).
      </p>
    ),
    source: 'ch. 6 §6.8, pp. 149, 152–153',
  },
  'ch06/6.8/q-warp': {
    kind: 'choice',
    id: 'ch06/6.8/q-warp',
    prompt: <>What do warp-level primitives let a block avoid?</>,
    options: [
      {
        label: 'Some block-wide barriers, and the round trip through shared memory for data exchanged inside a warp.',
        correct: true,
      },
      { label: 'All global memory traffic.', why: 'The data still comes from and goes to global memory. What changes is how threads of a warp share it.' },
      { label: 'Control divergence.', why: 'They do not change which path threads take; that is the divergence row.' },
    ],
    explain: (
      <p>
        Threads of a warp can hand values to each other directly and synchronise among themselves (chapter 4 named
        this; chapters 10 to 12 use it). The block still needs a barrier where warps exchange results. Porting note:
        the lane mask is 32 bits in CUDA and must be 64 bits in HIP, and Metal&rsquo;s SIMD-group functions take no
        mask (infra facts B3, C2).
      </p>
    ),
    source: 'ch. 6 §6.8, p. 152; ch. 4 §4.4, pp. 77–78',
  },
  'ch06/6.8/q-vector-memory': {
    kind: 'choice',
    id: 'ch06/6.8/q-vector-memory',
    prompt: <>Why does the table give vector loads the memory benefit &ldquo;more parallel memory accesses&rdquo;?</>,
    options: [
      {
        label:
          'Each instruction puts more bytes in flight, which covers DRAM latency when there are not enough warps to do it.',
        correct: true,
      },
      { label: 'Vector loads move fewer bytes.', why: 'The bytes are the same. Only the number of instructions that ask for them falls.' },
      { label: 'Vector loads avoid bank conflicts.', why: 'Banks belong to shared memory; this row is about global loads.' },
    ],
    explain: (
      <p>
        Section 6.2: the bus is kept busy only if enough requests are outstanding, and resident warps are one way to
        have them. Section 6.3, p. 139: when a kernel cannot reach full occupancy, wider loads are the other way. The
        same benefit appears in the occupancy row, from the other direction.
      </p>
    ),
    source: 'ch. 6 §6.2, pp. 135–136; §6.3, p. 139; Fig. 6.13, p. 148',
  },
};

// ---------------------------------------------------------------- the decision tool
const ROLE_CHIP: Record<'relieves' | 'spends', ReactNode> = {
  relieves: <span className="chip ok">relieves it</span>,
  spends: <span className="chip fail">spends what is short</span>,
};

function CostLine({ row }: { row: Row }) {
  return (
    <p style={{ margin: '0.35rem 0 0' }}>
      <b>Costs:</b> {row.cost.text}{' '}
      {row.cost.source ? <span className="muted">({row.cost.source})</span> : null}
    </p>
  );
}

function RowDetails({ row }: { row: Row }) {
  return (
    <div style={{ marginTop: '0.6rem', borderTop: '1px dashed var(--rule)', paddingTop: '0.6rem' }} data-details={row.id}>
      <table className="data" style={{ margin: 0 }}>
        <tbody>
          <tr>
            <th scope="row">Benefit to the cores</th>
            <td>{row.benefitCompute}</td>
          </tr>
          <tr>
            <th scope="row">Benefit to memory</th>
            <td>{row.benefitMemory ?? <span className="muted">none listed in the table</span>}</td>
          </tr>
          <tr>
            <th scope="row">Strategies</th>
            <td>
              {row.strategies.map((s, i) => (
                <div key={i}>{s}</div>
              ))}
            </td>
          </tr>
          <tr>
            <th scope="row">What the text adds</th>
            <td>
              {row.commentary} <span className="muted">(pp. {row.pages})</span>
            </td>
          </tr>
          <tr>
            <th scope="row">Returns in</th>
            <td>
              {row.returnsIn.map((r, i) => (
                <span key={r.ch}>
                  {i ? '; ' : ''}
                  <a href={href('ch' + String(r.ch).padStart(2, '0'))}>ch. {r.ch}</a> {r.what}
                </span>
              ))}
            </td>
          </tr>
          <tr>
            <th scope="row">Seen in real backends</th>
            <td>
              {row.seenIn.length ? (
                row.seenIn.map((s, i) => (
                  <div key={i} data-seen={s.fact}>
                    {s.absent ? <b>Not in </b> : null}
                    <code>{s.where}</code>: {s.what} <span className="muted">(facts {s.fact})</span>
                  </div>
                ))
              ) : (
                <span className="muted">not checked</span>
              )}
            </td>
          </tr>
          <tr>
            <th scope="row">Taught in</th>
            <td>
              {row.taught.length ? (
                row.taught.map(([c, s], i) => (
                  <span key={c + s}>
                    {i ? ', ' : ''}
                    <a href={href(c, s)}>{c === 'ch06' ? `§${s}` : `ch. ${Number(c.slice(2))} §${s}`}</a>
                  </span>
                ))
              ) : (
                <span className="muted">chapter 9 onwards</span>
              )}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function DecisionTool() {
  const [symptom, setSymptom] = useState<SymptomId | null>('global-traffic');
  const [open, setOpen] = useState<Set<string>>(new Set());
  const ranked = rankRows(symptom);
  const s = SYMPTOMS.find((x) => x.id === symptom);
  const toggle = (id: string) =>
    setOpen((o) => {
      const n = new Set(o);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  const allOpen = open.size === ROWS.length;
  return (
    <Bench
      title="Checklist as a decision tool"
      task="Pick what you observe. The rows that relieve it rise to the top; open a row for the rest of it."
      foot={
        <>
          The twelve rows of Fig. 6.13 in our wording, with the book&rsquo;s columns (category, benefit to the cores,
          benefit to memory, strategies) and three of ours: what it costs, where it returns, and where it is seen in
          real backends (from <code>notes/today/ch06-facts.md</code> D1 to D4 only). The symptoms and their ranking are
          ours; each link cites the page that supports it.
        </>
      }
    >
      <div className="row" role="radiogroup" aria-label="What you observe" style={{ marginBottom: '0.8rem' }}>
        {SYMPTOMS.map((x) => (
          <button
            key={x.id}
            className="quiet"
            aria-pressed={symptom === x.id}
            data-symptom={x.id}
            onClick={() => setSymptom(x.id)}
          >
            {x.label}
          </button>
        ))}
        <button className="quiet" aria-pressed={symptom === null} data-symptom="none" onClick={() => setSymptom(null)}>
          Book order
        </button>
      </div>
      <p className="status-line" data-looks>
        {s ? (
          <>
            <b>Looks like:</b> {s.looksLike} {s.relief.length} row{s.relief.length > 1 ? 's' : ''} relieve
            {s.relief.length > 1 ? '' : 's'} it
            {s.spends.length ? `; ${s.spends.length} would spend what is already short` : ''}.
          </>
        ) : (
          <>The table in the book&rsquo;s order: three rows for compute, six for memory, two for synchronisation, one general.</>
        )}
      </p>
      <div className="row" style={{ margin: '0.4rem 0 0.8rem' }}>
        <button className="quiet" onClick={() => setOpen(allOpen ? new Set() : new Set(ROWS.map((r) => r.id)))}>
          {allOpen ? 'Close all rows' : 'Open all rows'}
        </button>
      </div>
      <ol style={{ listStyle: 'none', padding: 0, margin: 0, display: 'grid', gap: '0.5rem' }}>
        {ranked.map(({ row, role, relief }) => (
          <li
            key={row.id}
            className="panel"
            data-row={row.id}
            data-role={role}
            style={{
              opacity: role === 'other' && symptom ? 0.72 : 1,
              borderLeft:
                role === 'relieves' ? '3px solid var(--ok)' : role === 'spends' ? '3px solid var(--fail)' : undefined,
            }}
          >
            <div className="row" style={{ justifyContent: 'space-between', gap: '0.4rem 1rem' }}>
              <div>
                <b>{row.name}</b> <span className="muted">· {CATEGORY_NAME[row.category]}</span>{' '}
                {role !== 'other' ? ROLE_CHIP[role] : null}
              </div>
              <button className="quiet" aria-expanded={open.has(row.id)} onClick={() => toggle(row.id)}>
                {open.has(row.id) ? 'Less' : 'More'}
              </button>
            </div>
            {relief ? (
              <p style={{ margin: '0.35rem 0 0' }} data-how>
                <b>{role === 'relieves' ? 'How:' : 'Why it hurts here:'}</b> {relief.how}{' '}
                <span className="muted">({relief.page})</span>
              </p>
            ) : null}
            <CostLine row={row} />
            {open.has(row.id) ? <RowDetails row={row} /> : null}
          </li>
        ))}
      </ol>
    </Bench>
  );
}

// ---------------------------------------------------------------- sorting drill
function SortDrill() {
  const [place, setPlace] = useState<Placement>({});
  const score = sortScore(place);
  const book = categoryCounts();
  return (
    <Bench
      title="Sort the twelve"
      task="File each optimisation under the category the book gives it. Each card says at once whether you agree with the book."
      foot={
        <>
          The book&rsquo;s rule (p. 149): a row that helps in several ways is filed under its primary intent. The
          reasons on the cards are ours.
        </>
      }
    >
      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(19rem, 1fr))' }}>
        {ROWS.map((r) => {
          const c = place[r.id];
          const verdict = c ? (c === r.category ? 'right' : 'wrong') : undefined;
          return (
            <div key={r.id} className="panel" data-card={r.id} data-verdict={verdict ?? 'open'}>
              <div>
                <b>{r.name}</b>
              </div>
              <div className="seg" style={{ display: 'flex', marginTop: '0.45rem', flexWrap: 'wrap' }}>
                {CATEGORIES.map((k) => (
                  <button
                    key={k}
                    aria-pressed={c === k}
                    data-cat={k}
                    onClick={() => setPlace((p) => ({ ...p, [r.id]: k }))}
                  >
                    {k === 'compute' ? 'Compute' : k === 'memory' ? 'Memory' : k === 'sync' ? 'Sync' : 'General'}
                  </button>
                ))}
              </div>
              {verdict ? (
                <p style={{ margin: '0.45rem 0 0' }}>
                  {verdict === 'right' ? (
                    <span className="t-ok">Same as the book.</span>
                  ) : (
                    <span className="t-fail">The book files it under {CATEGORY_NAME[r.category].toLowerCase()}.</span>
                  )}{' '}
                  {r.why}
                </p>
              ) : null}
            </div>
          );
        })}
      </div>
      <table className="data" style={{ marginTop: '1rem', marginBottom: '0.4rem' }} aria-label="Tally">
        <thead>
          <tr>
            <th>Category</th>
            <th>What it targets (p. 149)</th>
            <th className="num">Yours</th>
            <th className="num">The book&rsquo;s</th>
          </tr>
        </thead>
        <tbody>
          {CATEGORIES.map((k) => (
            <tr key={k} data-tally={k}>
              <td>{CATEGORY_NAME[k]}</td>
              <td>{CATEGORY_AIM[k]}</td>
              <td className="num">{score.tally[k].yours}</td>
              <td className="num">{score.done ? book[k] : '?'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="row">
        <p className="status-line" data-sort-result style={{ margin: 0 }}>
          {score.done
            ? `All twelve placed. ${score.right} of 12 where the book puts them. The book's count: ${countLine(book)} = 12.`
            : `${score.placed} of 12 placed.`}
        </p>
        <button className="quiet" onClick={() => setPlace({})} disabled={!score.placed}>
          Start again
        </button>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- scenario cards
const OUTCOMES: Outcome[] = ['faster', 'same', 'slower'];
const OUTCOME_CODE: Record<Outcome, number> = { faster: 0, same: 1, slower: 2 };

function Scenarios() {
  const { progress, answer } = useProgress();
  const picks = SCENARIOS.map((s) => {
    const v = progress.answers[`ch06/6.8/p-${s.id}`]?.value;
    return typeof v === 'number' ? OUTCOMES[v] : undefined;
  });
  const done = picks.filter(Boolean).length;
  const right = SCENARIOS.filter((s, i) => picks[i] === s.expect).length;
  return (
    <Bench
      title="Scenario cards"
      task="For each proposal, predict: faster, about the same, or slower. Then read why."
      foot={
        <>
          Your first prediction for each card is kept with your progress. The cases are ours; each is decided by the
          rule of p. 153 and the page given with it.
        </>
      }
    >
      <div style={{ display: 'grid', gap: '0.7rem' }}>
        {SCENARIOS.map((s, i) => {
          const pick = picks[i];
          const row = rowById(s.row);
          return (
            <div key={s.id} className="panel" data-scenario={s.id} data-picked={pick ?? 'none'}>
              <p style={{ margin: '0 0 0.3rem' }}>
                <b>Card {i + 1}.</b> {s.situation}
              </p>
              <p style={{ margin: '0 0 0.5rem' }}>
                <b>Proposal:</b> {s.change}
              </p>
              <div className="seg">
                {OUTCOMES.map((o) => (
                  <button
                    key={o}
                    aria-pressed={pick === o}
                    data-outcome={o}
                    disabled={!!pick}
                    onClick={() => answer(`ch06/6.8/p-${s.id}`, OUTCOME_CODE[o], o === s.expect)}
                  >
                    {OUTCOME_LABEL[o]}
                  </button>
                ))}
              </div>
              {pick ? (
                <div style={{ marginTop: '0.5rem' }} data-reveal>
                  <p style={{ margin: '0 0 0.3rem' }}>
                    {pick === s.expect ? (
                      <span className="t-ok">
                        <b>{OUTCOME_LABEL[s.expect]}, as you predicted.</b>
                      </span>
                    ) : (
                      <span className="t-fail">
                        <b>
                          You predicted {OUTCOME_LABEL[pick].toLowerCase()}; the rule says{' '}
                          {OUTCOME_LABEL[s.expect].toLowerCase()}.
                        </b>
                      </span>
                    )}
                  </p>
                  <p style={{ margin: 0 }}>
                    {s.why} <span className="muted">Row: {row.name}. Source: {s.page}.</span>
                  </p>
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
      <p className="status-line" data-scenario-result style={{ margin: '0.8rem 0 0' }}>
        {done === SCENARIOS.length
          ? `${right} of ${SCENARIOS.length} predicted as the rule says.`
          : `${done} of ${SCENARIOS.length} predicted.`}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the page
export function S68() {
  const q = questionsS68;
  return (
    <>
      <p>
        The chapter closes Part 1 by gathering what chapters 4, 5 and 6 taught into one table of twelve
        optimisations. It does not claim to be complete. These are the ones worth considering first for almost any
        kernel; later chapters add their own, such as constant memory in chapter 7. For each row the table gives
        what it does for the cores, what it does for memory, and how to do it. Parts 2 and 3 of the book refer back
        to it constantly.
      </p>
      <Source section="6.8" pages="147–149, 153" figs="Fig. 6.13" />

      <h2>Four categories, twelve rows</h2>
      <p>
        The book sorts the rows by the aspect of performance each one targets: keeping the cores busy, using the
        memory system well, or waiting less at barriers. Many rows help in more than one way; each is filed under its
        primary intent. One row gets a category of its own. Sort the twelve yourself before the table below shows
        you the book&rsquo;s answer.
      </p>
      <Source section="6.8" pages="149" />
      <SortDrill />
      <Ask q={q['ch06/6.8/q-general']} />

      <h2>From a symptom to a row</h2>
      <p>
        Read as a table, the checklist answers &ldquo;what does tiling do?&rdquo;. The question you will actually
        have runs the other way: &ldquo;my kernel waits on this; what do I try?&rdquo;. So here the table starts from
        a symptom. Every row also carries a cost, because section 6.9&rsquo;s whole argument is that each
        optimisation spends something.
      </p>
      <Beyond>
        <p>
          The symptom list, the ranking, the cost wording where it says &ldquo;not stated in the book&rdquo;, and the
          &ldquo;seen in&rdquo; column are ours. Costs the chapter states are cited to their page; one vendor cost
          (vector loads) to <code>ch06-facts.md</code> A11.
        </p>
      </Beyond>
      <DecisionTool />

      <p>
        Three things to notice while you click. Half the rows relieve more than one symptom, so the same optimisation
        keeps coming back for different reasons. Coarsening relieves six of the eight symptoms, which is what its
        &ldquo;general&rdquo; label means in practice. And for three symptoms a row is marked in red: it would spend
        exactly what the kernel is short of.
      </p>
      <Ask q={q['ch06/6.8/q-sectors']} />
      <Ask q={q['ch06/6.8/q-vector-memory']} />
      <Ask q={q['ch06/6.8/q-warp']} />

      <p>
        The table summarises pages 149 to 153, and in two rows the summary and the text do not quite agree. Where
        they differ, go by the text and expect the later chapter to settle it.
      </p>
      <Aside>
        <p>
          <b>Table and text differ.</b> Register tiling: the table says data moves once between shared memory and
          registers; the text (p. 151) says between global or shared memory and registers. Warp-level primitives: the
          table claims less global and shared traffic; the text (p. 152) mentions only avoiding shared memory, and
          does not say why global traffic would fall.
        </p>
      </Aside>

      <h2>Predict before you apply</h2>
      <p>
        A row says what an optimisation can relieve. Whether it speeds up your kernel depends on whether that is what
        limits the kernel, and on whether the resource it spends was already short. That is the rule of section{' '}
        <a href={href('ch06', '6.9')}>6.9</a>, stated on p. 153. Five proposals; decide each one before you look.
      </p>
      <Source section="6.9" pages="153" />
      <Scenarios />
      <Ask q={q['ch06/6.8/q-tiling-occ']} />

      <h2>Where you will meet these rows</h2>
      <p>
        When you read the GPU backends of PyTorch, MLX or llama.cpp, the rows are easy to spot once you know their
        shapes. PyTorch&rsquo;s elementwise kernels use vector loads and choose the width per tensor from the
        pointer&rsquo;s alignment: the scalar fallback the book leaves to exercise 4, done at run time. llama.cpp pads
        its quantised-matmul tiles in shared memory against bank conflicts, and its flash-attention kernel loads the
        next block into a second buffer while computing on the first. MLX pads its Metal GEMM tiles too, but its main
        loop is single-buffered with two barriers per step: the &ldquo;before&rdquo; loop of 6.7, in production
        code. The &ldquo;seen in&rdquo; line of each row above names the files.
      </p>
      <Today checked="2026-11-20">
        <p>
          Read on the projects&rsquo; default branches (<code>notes/today/ch06-facts.md</code> D1 to D4). File details
          drift; the facts file dates each one. A row marked &ldquo;not checked&rdquo; was not looked for, which is
          not the same as absent.
        </p>
      </Today>
    </>
  );
}
