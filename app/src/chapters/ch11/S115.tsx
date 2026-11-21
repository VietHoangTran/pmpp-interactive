import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Beyond, Bench, Simplification, Source, Today } from '../../components/notes';
import { useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  CHARGE_LABEL,
  CLOSED_FORM_TEXT,
  VOCAB_POW2,
  VOCAB_QWEN3_8B,
  bookKoggeStoneSteps,
  breakEven,
  closedForm,
  lg,
  slotsTotal,
  span,
  stepsWithUnits,
  type Charge,
  type NetId,
} from '../../sim/prefixnet';
import { widthChoices } from '../../sim/warps';

/** A caption above a group of buttons (a <label> would forward clicks to the first button). */
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const WRAP: CSSProperties = { flexWrap: 'wrap' };
const int = (v: number) => Math.round(v).toLocaleString('en-US');
const dec = (v: number, d = 1) => Number(v.toFixed(d)).toLocaleString('en-US');
const times = (v: number) => (v >= 100 ? int(v) : v >= 10 ? dec(v, 1) : dec(v, 2));

// ---------------------------------------------------------------- the explorer
const ROWS: NetId[] = ['sequential', 'kogge-stone', 'brent-kung'];
const ROW_NAME: Record<NetId, string> = { sequential: 'Loop (Fig. 11.1)', 'kogge-stone': 'Kogge-Stone', 'brent-kung': 'Brent-Kung' };
const CHARGES: Charge[] = ['threads', 'warps', 'adds'];
const CHARGE_NOTE: Record<Charge, string> = {
  threads:
    'Every thread of the block is charged in every step, whether it adds or not: the book\'s estimate, (N·log₂N)/P for Kogge-Stone (p. 266). Brent-Kung\'s block has N/2 threads (our kernel in section 11.10).',
  warps:
    'A warp with at least one adding lane is charged all its lanes; a warp with none is free. This is the reason the book gives on p. 266 for resource use above the formula.',
  adds: 'Only the threads that add are charged: the cost the closed forms count.',
};
/** Slider position (0..120) to execution units, on a log scale: 2^(v/10). */
const unitsOf = (v: number) => Math.max(1, Math.round(2 ** (v / 10)));
const sliderOf = (p: number) => Math.round(10 * Math.log2(p));
const PRESETS = [
  { key: 'b512', label: 'Book: 512 elements', logN: 9, p: 4096, charge: 'threads' as Charge },
  { key: 'b1024', label: 'Book: 1,024 elements, 32 units', logN: 10, p: 32, charge: 'threads' as Charge },
  { key: 'vocab', label: 'A vocabulary row: 2¹⁷ elements, 1,024 units', logN: 17, p: 1024, charge: 'threads' as Charge },
];

function Bar({ value, max }: { value: number; max: number }) {
  const w = max > 0 ? Math.max(1, (value / max) * 100) : 0;
  return (
    <span aria-hidden="true" style={{ display: 'block', height: 6, marginTop: 3, background: 'var(--rule-soft)', borderRadius: 2 }}>
      <span style={{ display: 'block', height: 6, width: `${w}%`, background: 'var(--ink-2)', borderRadius: 2 }} />
    </span>
  );
}

function WorkExplorer() {
  const { reports } = useReports();
  const widths = useMemo(() => widthChoices(reports as Reports), [reports]);
  const [logN, setLogN] = useState(10);
  const [p, setP] = useState(32);
  const [pText, setPText] = useState('32');
  const [charge, setCharge] = useState<Charge>('threads');
  const [wKey, setWKey] = useState('book');
  const W = (widths.find((c) => c.key === wKey) ?? widths[0]).width;
  const n = 2 ** logN;
  const setUnits = (v: number) => {
    const c = Math.max(1, Math.min(4096, Math.round(v)));
    setP(c);
    setPText(String(c));
  };

  const rows = ROWS.map((id) => {
    const adds = closedForm(id, n);
    const steps = stepsWithUnits(id, n, p, charge, W);
    return { id, adds, steps, unlimited: span(id, n) };
  });
  const loop = rows[0];
  const maxAdds = Math.max(...rows.map((r) => r.adds));
  const maxSteps = Math.max(...rows.map((r) => r.steps));
  const beKS = breakEven('kogge-stone', n, charge, W);
  const beBK = breakEven('brent-kung', n, charge, W);
  const naive = closedForm('naive', n);
  const ks = rows[1];

  return (
    <Bench
      title="Work and steps as N grows"
      task="Start at the book's 1,024 elements on 32 units. Then shrink P until Kogge-Stone loses to the loop, and grow N."
      foot={
        <>
          Additions are counted from the networks of section 11.2 (Kogge-Stone) and section 11.10 (Brent-Kung) and agree
          with the closed forms beside them. A step with more charged lanes than units takes several rounds: it costs
          ⌈charge / P⌉. The loop is the sequential reference: about N steps, as the book counts it, whatever P is.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <span>
            N = <b data-value="n">{int(n)}</b> elements
          </span>
          <input type="range" min={3} max={20} value={logN} aria-label="Elements, as a power of two" onChange={(e) => setLogN(Number(e.target.value))} style={{ width: '13rem' }} />
        </label>
        <label className="field">
          <span>
            P = <b data-value="p">{int(p)}</b> execution units
          </span>
          <input type="range" min={0} max={120} value={sliderOf(p)} aria-label="Execution units" onChange={(e) => setUnits(unitsOf(Number(e.target.value)))} style={{ width: '13rem' }} />
        </label>
        <label className="field">
          exactly
          <input type="number" min={1} max={4096} step={1} value={pText} aria-label="Execution units, exactly" style={{ width: '6rem' }}
            onChange={(e) => {
              setPText(e.target.value);
              const v = Number(e.target.value);
              if (e.target.value.trim() !== '' && Number.isFinite(v) && v >= 1) setP(Math.min(4096, Math.round(v)));
            }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Charge each step for
          <span className="seg" role="group" aria-label="Charge each step for" style={WRAP}>
            {CHARGES.map((c) => (
              <button key={c} aria-pressed={charge === c} onClick={() => setCharge(c)}>{CHARGE_LABEL[c]}</button>
            ))}
          </span>
        </span>
        <label className="field">
          Warp width
          <select value={wKey} aria-label="warp width" disabled={charge !== 'warps'} onChange={(e) => setWKey(e.target.value)}>
            {widths.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span style={FIELD}>
          Presets
          <span className="seg" role="group" aria-label="Presets" style={WRAP}>
            {PRESETS.map((q) => (
              <button key={q.key} onClick={() => { setLogN(q.logN); setUnits(q.p); setCharge(q.charge); }}>{q.label}</button>
            ))}
          </span>
        </span>
      </div>
      <p className="muted" style={{ margin: '0 0 0.7rem' }} data-readout="charge">{CHARGE_NOTE[charge]}</p>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0, minWidth: '30rem' }} data-part="work-table">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>Scan</th>
              <th className="num">additions</th>
              <th className="num">× the loop's</th>
              <th className="num">steps with P units</th>
              <th className="num">against the loop</th>
              <th className="num">unlimited units</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} data-row={r.id} data-adds={r.adds} data-steps={r.steps} data-span={r.unlimited}>
                <td style={{ paddingLeft: '0.6rem' }}>{ROW_NAME[r.id]}</td>
                <td className="num">
                  {int(r.adds)}
                  <Bar value={r.adds} max={maxAdds} />
                  <span className="muted" style={{ display: 'block', fontSize: '0.8em', whiteSpace: 'nowrap' }}>{CLOSED_FORM_TEXT[r.id]}</span>
                </td>
                <td className="num" data-cell="work-ratio">{times(r.adds / loop.adds)}</td>
                <td className="num">
                  {int(r.steps)}
                  <Bar value={r.steps} max={maxSteps} />
                </td>
                <td className="num" data-cell="vs-loop">
                  {r.id === 'sequential' ? '' : r.steps < loop.steps ? (
                    <span className="chip ok">{times(loop.steps / r.steps)}× fewer</span>
                  ) : r.steps === loop.steps ? (
                    <span className="chip">as many</span>
                  ) : (
                    <span className="chip fail">{times(r.steps / loop.steps)}× more</span>
                  )}
                </td>
                <td className="num">{int(r.unlimited)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="cols" style={{ marginTop: '0.9rem' }}>
        <div className="panel" aria-live="polite" data-readout="breakeven" data-ks={beKS ?? ''} data-bk={beBK ?? ''}>
          <p style={{ margin: 0 }}>
            Kogge-Stone needs fewer steps than the loop from <b data-value="be-ks">{beKS == null ? 'no' : int(beKS)}</b>{' '}
            execution units; Brent-Kung from <b data-value="be-bk">{beBK == null ? 'no' : int(beBK)}</b>.
          </p>
          <p className="muted" style={{ margin: '0.4rem 0 0' }}>
            The book's estimate for Kogge-Stone, (N·log₂N)/P = <span data-value="book-ks">{dec(bookKoggeStoneSteps(n, p), 1)}</span>{' '}
            steps, beats the loop once P &gt; log₂N = {lg(n)}. Rounding each step up to whole rounds can push the break-even
            a unit higher.
          </p>
        </div>
        <div className="panel" data-readout="extremes">
          <p style={{ margin: 0 }}>
            With unlimited units Kogge-Stone needs N/log₂N = <b data-value="unlimited-ratio">{dec(loop.unlimited / ks.unlimited, 1)}</b>{' '}
            times fewer steps than the loop.
          </p>
          <p className="muted" style={{ margin: '0.4rem 0 0' }}>
            Lane slots charged in all: Kogge-Stone {int(slotsTotal('kogge-stone', n, charge, W))}, Brent-Kung{' '}
            {int(slotsTotal('brent-kung', n, charge, W))}. The naive scan of section 11.2 would do N(N − 1)/2 ={' '}
            <span data-value="naive">{int(naive)}</span> additions, {times(naive / loop.adds)} times the loop's.
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const KS512 = closedForm('kogge-stone', 512);
const LOOP512 = closedForm('sequential', 512);
const STEPS2048 = stepsWithUnits('kogge-stone', 2048, 64, 'threads');
const BE4096 = breakEven('kogge-stone', 4096, 'threads')!;
const STEPS1024 = stepsWithUnits('kogge-stone', 1024, 32, 'threads');

export const questionsS115: Record<string, Question> = {
  'ch11/11.5/q-work512': {
    kind: 'numeric',
    id: 'ch11/11.5/q-work512',
    prompt: <>How many additions does Kogge-Stone perform on a segment of 512 elements?</>,
    answer: KS512,
    unit: 'additions',
    explain: (
      <p>
        log₂512 = 9 steps. Summing N − stride over the strides 1, 2, …, 256 gives 512·9 − (1 + 2 + … + 256) = 4608 − 511 ={' '}
        {int(KS512)}. The loop does {LOOP512}.
      </p>
    ),
    source: 'ch. 11 §11.5, p. 266',
  },
  'ch11/11.5/q-ratio': {
    kind: 'numeric',
    id: 'ch11/11.5/q-ratio',
    prompt: <>How many times the sequential loop's work is that? Give two decimals.</>,
    answer: Number((KS512 / LOOP512).toFixed(2)),
    tol: 0.01,
    unit: 'times',
    explain: (
      <p>
        {KS512} / {LOOP512} = {(KS512 / LOOP512).toFixed(3)}: the book's "between eight and nine times". The ratio is about
        log₂N − 1, so it keeps growing with the segment.
      </p>
    ),
    source: 'ch. 11 §11.5, p. 266',
  },
  'ch11/11.5/q-steps': {
    kind: 'numeric',
    id: 'ch11/11.5/q-steps',
    prompt: <>By the book's model, how many steps does Kogge-Stone need for 2,048 elements on 64 execution units?</>,
    answer: STEPS2048,
    unit: 'steps',
    explain: (
      <p>
        (N·log₂N)/P = 2048·11/64 = {STEPS2048}, against the loop's 2,048: a {(2048 / STEPS2048).toFixed(1)}-fold reduction.
        Twice the units of the book's example, twice the elements, and the gain grows only from 3.2 to {(2048 / STEPS2048).toFixed(1)}.
      </p>
    ),
    source: 'ch. 11 §11.5, p. 266 (the numbers are ours)',
  },
  'ch11/11.5/q-breakeven': {
    kind: 'numeric',
    id: 'ch11/11.5/q-breakeven',
    prompt: (
      <>
        For 4,096 elements, what is the smallest number of execution units P with which Kogge-Stone needs fewer steps than
        the loop, by the book's model?
      </>
    ),
    answer: BE4096,
    unit: 'units',
    explain: (
      <p>
        (4096·12)/P &lt; 4096 holds when P &gt; 12, so {BE4096}. With 12 units the two tie; with fewer the parallel scan is
        slower than one thread looping. The threshold is log₂N, so it rises with the data size, which is why the book's
        summary says Kogge-Stone suits modest-sized blocks on processors with plenty of execution units.
      </p>
    ),
    source: 'ch. 11 §11.5, pp. 266–267; §11.11, p. 285',
  },
  'ch11/11.5/q-energy': {
    kind: 'choice',
    id: 'ch11/11.5/q-energy',
    prompt: <>On some device a Kogge-Stone scan finishes sooner than the loop. Is its extra work then free?</>,
    options: [
      { label: 'Yes: once it is faster, only the time counts.', why: 'Time is one of two costs the book names.' },
      { label: 'No: each extra addition still costs energy, which matters on phones.', correct: true },
      { label: 'Yes, as long as the extra additions run on units that would otherwise idle.', why: 'Executing an addition costs energy whichever unit runs it; idle units are not what makes it free.' },
    ],
    explain: (
      <p>
        The book names two prices. When P is small the parallel scan needs more steps than the loop. And even when it is
        faster, the extra work consumes additional energy, which makes the kernel less suitable for power-constrained
        settings such as mobile applications.
      </p>
    ),
    source: 'ch. 11 §11.5, p. 267',
  },
  'ch11/11.5/q-slots': {
    kind: 'choice',
    id: 'ch11/11.5/q-slots',
    prompt: <>Why does the book say Kogge-Stone's real resource use is closer to N·log₂N than to N·log₂N − (N − 1)?</>,
    options: [
      { label: 'The formula forgets the additions of the first step.', why: 'The first step, stride 1, is in the sum: N − 1 additions.' },
      { label: 'Idle threads still occupy lanes of warps that keep executing.', correct: true },
      { label: 'Each addition needs two shared-memory reads, which doubles the work.', why: 'The count is of additions, not of memory accesses.' },
    ],
    explain: (
      <p>
        The N − 1 subtracted are the threads below the stride, which sit out their step. They do not leave the warp: a warp
        executes for all its lanes, with the inactive ones masked off, until the whole warp is done. Their slots are spent
        either way. Section 11.10 turns the same fact against Brent-Kung.
      </p>
    ),
    source: 'ch. 11 §11.5, p. 266; NVIDIA and AMD on masked lanes: notes/today/ch04-facts.md A9, B7',
  },
};

// ---------------------------------------------------------------- the section
export function S115() {
  const q = questionsS115;
  return (
    <>
      <p>
        Sections 11.2 to 11.4 made Kogge-Stone fast by giving it a short span and then cutting its synchronisation. This
        section counts what it costs in additions, and finds the conditions under which that cost makes the parallel scan
        slower than the six-line loop it replaces.
      </p>

      <h2>Work efficiency</h2>
      <p>
        An algorithm's <b>work efficiency</b> is how close the work it does comes to the minimum the computation needs,
        where "minimum" means the least work of any known algorithm. For an inclusive scan the minimum is N − 1 additions,
        what the loop of Fig. 11.1 does. The naive parallel scan of section 11.2, one thread per output summing its prefix
        from scratch, does N(N − 1)/2: O(N²). That is not work efficient. For 1,024 elements it is 523,776 additions where
        1,023 would do.
      </p>
      <Source section="11.5" pages="265" />

      <h2>Counting Kogge-Stone's additions</h2>
      <p>
        Kogge-Stone runs log₂N steps. In the step with stride s, the s threads below the stride sit out and the other N − s
        add once each. Sum that over s = 1, 2, 4, …, N/2. The N terms give N·log₂N; the strides are the series 1 + 2 + … +
        N/2 = N − 1. Kogge-Stone does N·log₂N − (N − 1) additions, which is O(N·log₂N).
      </p>
      <Source section="11.5" pages="266" figs="Fig. 11.2" />
      <Ask q={q['ch11/11.5/q-work512']} label="Work it out" />
      <Ask q={q['ch11/11.5/q-ratio']} label="Work it out" />
      <p>
        That is far better than the naive scan's O(N²) and still worse than the loop's O(N). The gap is not small even for
        modest segments: at 512 elements, the book's example, Kogge-Stone does between eight and nine times the loop's work,
        and the ratio grows by about one with every doubling of N.
      </p>

      <h2>On hardware, closer to N·log₂N</h2>
      <p>
        The formula charges only the threads that add. On a real device, the book points out, threads that have stopped
        adding still use execution resources until the warp they belong to completes. So the resources Kogge-Stone consumes
        are nearer N·log₂N than N·log₂N − (N − 1). Either way the order is O(N·log₂N).
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A9, B7, C4)">
        <p>
          NVIDIA: lanes not on the current path "will be masked off while the threads which follow the branch are executed".
          AMD: "ALUs still process the full warp, but the result for divergent threads is masked out". Apple documents no
          execution model for inactive SIMD-group threads.
        </p>
      </Today>
      <Ask q={q['ch11/11.5/q-slots']} />

      <h2>Fewer steps, if you have the units</h2>
      <p>
        The extra work buys a short span. The loop runs N iterations; the kernel's loop runs at most log₂N. With unlimited
        execution units Kogge-Stone needs about N/log₂N times fewer steps than the loop: 512/9 = 56.9 at 512 elements.
        Units are never unlimited. With P execution units the book estimates Kogge-Stone at (N·log₂N)/P steps against the
        loop's N. If P = N that is log₂N again; with fewer units every step takes several rounds.
      </p>
      <Source section="11.5" pages="266" />
      <Predict
        id="ch11/11.5/p-steps"
        prompt={<>With 1,024 elements and 32 execution units, how many times fewer steps does Kogge-Stone take than the loop?</>}
        unit="times fewer"
      >
        {(v) => (
          <p>
            By the book's model, (1024·10)/32 = <b>{STEPS1024}</b> steps against the loop's 1,024:{' '}
            <b>{1024 / STEPS1024}</b> times fewer.{' '}
            {Math.abs(v - 1024 / STEPS1024) < 0.3
              ? 'That is what you predicted.'
              : v > 1024 / STEPS1024
                ? 'Less than you expected: 32 units against 1,024 threads means every one of the 10 steps takes 32 rounds.'
                : 'More than you expected: even at 32 rounds a step, 10 steps are fewer than 1,024.'}{' '}
            The 32 units do not give 32 times fewer steps; the extra work eats a factor of ten of them.
          </p>
        )}
      </Predict>
      <p>
        Now explore it. The explorer counts additions and steps for the loop, Kogge-Stone and Brent-Kung (section 11.10)
        across sizes from 8 to a million elements. The default charge is the book's: every thread of the block in every
        step. The other two charges show how much of that is the idle threads.
      </p>
      <Simplification>
        <p>
          One addition per unit per round, nothing else: no memory, no barriers, no latency, and a step starts only when
          the previous one has finished. A "unit" is the book's abstract execution unit, not a figure for any device.
        </p>
      </Simplification>
      <WorkExplorer />

      <h2>Two prices for the extra work</h2>
      <p>
        The book names two. First, the hardware does more for the same result, and if P is small the parallel scan needs
        more steps than the loop and is slower. Second, even when it is faster, the extra additions consume additional
        energy, which makes it a poor fit where power is limited, as on a phone. The chapter's summary turns the first into
        a rule: the units Kogge-Stone needs to break even with the loop grow with the data size, so it suits modest-sized
        blocks on processors with plenty of execution units.
      </p>
      <Source section="11.5" pages="267" figs="§11.11, p. 285" />
      <Ask q={q['ch11/11.5/q-steps']} />
      <Ask q={q['ch11/11.5/q-breakeven']} />
      <Ask q={q['ch11/11.5/q-energy']} />

      <h2>What comes next</h2>
      <p>
        The chapter gives two answers. <a href={href('ch11', '11.10')}>Section 11.10</a> presents Brent-Kung, which does 2N
        − 2 − log₂N additions, close to 2N, in about twice the steps; you can already compare it in the explorer. The book's
        real answer comes first, in <a href={href('ch11', '11.6')}>section 11.6</a>: let each thread scan a run of elements
        sequentially, which is work efficient, and keep the parallel scan for the few values left over.
      </p>
      <p>
        A sampler's cumulative sum runs over one vocabulary row, and Qwen3-8B's has {int(VOCAB_QWEN3_8B)} entries. Pick the
        preset of 2¹⁷ = {int(VOCAB_POW2)} elements on 1,024 units. Kogge-Stone still needs far fewer steps than one thread
        looping, but each of its 17 steps touches the whole row: {lg(VOCAB_POW2) - 1} or so times the loop's additions. That
        is the regime of many more elements than units where section 11.6's coarsening pays. llama.cpp's CUDA and HIP
        kernel and MLX's Metal kernel both give each thread a run of 4 values (for 4-byte types, in MLX) to scan on its own
        before any step between lanes, and walk a long row in chunks with a carried total.
      </p>
      <Beyond>
        <p>
          The sampling bridge is ours. Vocabulary size: notes/today/ch09-facts.md D7. llama.cpp's <code>cumsum_kernel</code>{' '}
          and MLX's <code>contiguous_scan</code>, read from source on 2026-11-20: ch11-facts D1, D3.
        </p>
      </Beyond>
    </>
  );
}
