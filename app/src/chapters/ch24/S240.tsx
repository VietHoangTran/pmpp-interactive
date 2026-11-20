import { useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Ask, Predict, type ChoiceQ, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { PLAN, useAllChapters } from '../registry';
import { GOALS, STEPS, goal, lanes, rollup, stepOf, stepsOf, type GoalId, type Step } from '../../sim/booksteps';
import {
  BOOK,
  DEFAULTS,
  DEVICES,
  LIMITS,
  LIMIT_ROWS,
  VENDOR,
  auditPrinted,
  built,
  figures,
  fmtInt,
  fmtX,
  marks,
  nvlinkOverHostLink,
  printedCells,
  ratio,
  regime,
  threshold,
  tileFor,
  type Device,
  type FigureSet,
  type LimitId,
  type Options,
  type RowId,
} from '../../sim/hwtable';

const P_HOSTLINK = 'ch24/start/p-hostlink';
const P_KNEE = 'ch24/start/p-knee';

// ---------------------------------------------------------------- 1. the steps map
// What each goal's set of chapters shows, in one or two sentences (PLAN.md sections 2, 3 and 5).
const GOAL_NOTE: Record<GoalId, string> = {
  matmul:
    'All three steps. Step one writes the kernel and tunes it, chapter 15 in step two takes it towards a library-grade one, and chapter 19 in step three shows how a framework lowers other work onto it.',
  attention:
    'Step two supplies the softmax, which is a reduction. Step three supplies the attention kernels and the batching argument.',
  sampling:
    'Every book chapter here is a step-two pattern. The book never assembles them into a sampler; module X1 does.',
  quantised:
    "The book offers two paragraphs on lower-precision formats in appendix A and chapter 15's intensity analysis. The kernels themselves are ours, in module X2.",
  backends:
    "Appendix C gives the book's memory models, which we set against unified memory, and module X3 reads real HIP and Metal backends. Every lab counts as well: one .cu source serves CUDA and HIP, and Metal gets its own kernel wherever it can express the lab.",
  reading:
    'Chapter 19 shows how a framework hands work to a library. Module X3 reads llama.cpp, MLX and PyTorch.',
};
const STEP_WORD: Record<string, string> = { 1: 'step one', 2: 'step two', 3: 'step three', outside: "outside the book's steps" };

function StepsMap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const chapters = useAllChapters();
  const [goalId, setGoalId] = useState<GoalId>('matmul');
  const g = goal(goalId);
  const lit = new Set(g.chapters);
  const record = useMemo(() => rollup(PLAN, progress.sections, reports), [progress.sections, reports]);
  const builtCh = new Map((chapters ?? []).filter((c) => c.sections.length).map((c) => [c.id, c]));
  const row = (id: string) => PLAN.find((r) => r.id === id)!;
  const nums = (step: Step) => g.chapters.filter((c) => stepOf(c) === step).map((c) => row(c).num);

  return (
    <Bench
      title="The book's three steps"
      task="Pick one of your goals. Which steps does it draw on?"
      foot="Steps as the book assigns them on pp. 577–578. Which chapters feed a goal comes from this course's plan, not from the book. A chapter either feeds a goal or does not; the map says nothing about how much of it you need. Your record counts sections you marked done and labs whose work/ copy passes on a real backend."
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span className="muted">Goal</span>
        <span className="seg" style={{ flexWrap: 'wrap' }} data-goals>
          {GOALS.map((x) => (
            <button key={x.id} aria-pressed={goalId === x.id} onClick={() => setGoalId(x.id)}>
              {x.label}
            </button>
          ))}
        </span>
      </div>
      {lanes(PLAN).map(({ step, rows }) => {
        const s = STEPS.find((x) => x.step === step)!;
        return (
          <div
            key={String(step)}
            className="cols"
            data-lane={String(step)}
            style={{ gridTemplateColumns: 'minmax(0, 13rem) minmax(0, 1fr)', gap: '0.5rem 1rem', padding: '0.6rem 0', borderTop: '1px solid var(--rule)' }}
          >
            <div>
              <b>{s.name}</b>
              <div className="muted" style={{ fontSize: '0.76rem' }}>
                {s.chapters}
              </div>
              <div className="muted" style={{ fontSize: '0.76rem', marginTop: '0.2rem' }}>
                {s.what}
              </div>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
              {rows.map((r) => {
                const on = lit.has(r.id!);
                const ch = builtCh.get(r.id!);
                const rec = record[r.id!];
                const chip: CSSProperties = {
                  flex: '1 1 8.6rem',
                  maxWidth: '12.5rem',
                  padding: '0.35rem 0.5rem 0.4rem',
                  borderRadius: 4,
                  border: `1px solid ${on ? 'var(--ink)' : 'var(--rule)'}`,
                  boxShadow: on ? 'inset 0 3px 0 var(--ink)' : undefined,
                  background: on ? 'var(--page)' : 'transparent',
                  color: on ? 'var(--ink)' : 'var(--ink-3)',
                  fontSize: '0.78rem',
                  lineHeight: 1.3,
                };
                return (
                  <div key={r.id} data-chip={r.id} data-lit={on ? 'true' : 'false'} style={chip}>
                    <div>
                      <b>{r.num}</b> {ch ? <a href={href(ch.id, ch.sections[0].id)}>{r.title}</a> : r.title}
                      {r.ours ? <span className="muted"> (ours)</span> : null}
                    </div>
                    {on && r.milestone ? <div style={{ fontWeight: 600, marginTop: '0.15rem' }}>{r.milestone}</div> : null}
                    <div style={{ fontSize: '0.7rem', marginTop: '0.15rem', color: on ? 'var(--ink-2)' : 'var(--ink-3)' }} data-record>
                      {ch ? (
                        <>
                          {rec.sectionsDone} of {ch.sections.length} sections done
                          {rec.labs ? `, ${rec.labsPassing} of ${rec.labs} labs passing` : ''}
                        </>
                      ) : (
                        'planned'
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
      <div className="panel" style={{ marginTop: '0.6rem' }} data-readout="goal" aria-live="polite">
        <p style={{ margin: '0 0 0.4rem' }}>
          <b>{g.label}</b> draws on {g.chapters.length} chapters:{' '}
          {stepsOf(goalId)
            .map((s) => `${STEP_WORD[String(s)]} (${nums(s).join(', ')})`)
            .join('; ')}
          {g.everyLab ? '; and every lab' : ''}.
        </p>
        <p style={{ margin: '0 0 0.5rem' }}>{GOAL_NOTE[goalId]}</p>
        <table className="data" style={{ margin: 0 }}>
          <tbody>
            {g.chapters.map((c) => (
              <tr key={c}>
                <td className="nowrap">
                  <b>{row(c).num}</b>
                </td>
                <td>{row(c).title}</td>
                <td>{row(c).why}</td>
                <td>{row(c).milestone ?? ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- 2. what limits what
// One line per wrong placement. The right row and its reason live in the model (LIMITS).
const NOT_THIS: Record<string, Partial<Record<LimitId, string>>> = {
  'ch24/start/q-limit-load': {
    compute: 'No arithmetic is done on the way. The bytes only move.',
    membw: 'Device memory can take the bytes faster than they arrive. Ask which path they arrive on.',
    memcap: 'Capacity decides whether the weights fit, not how long the copy takes.',
    gpulink: 'The copy starts in host memory, so it does not travel between GPUs.',
  },
  'ch24/start/q-limit-decode': {
    compute: 'Two operations per weight fetched leave the arithmetic units waiting. Ask what they wait for.',
    memcap: 'Capacity decides whether the model fits. Once it does, something else sets the time per token.',
    hostlink: 'The weights and the KV cache are already on the device. Nothing crosses to the host per weight.',
    gpulink: 'One GPU is doing this.',
  },
  'ch24/start/q-limit-prefill': {
    membw: 'A matrix-matrix product reuses each operand for a whole row or column of outputs, so a well-written kernel does not wait on memory.',
    memcap: 'Capacity decides whether the prompt fits, not how fast the products run.',
    hostlink: "The prompt's tokens cross once. The products then run on the device.",
    gpulink: 'One GPU is doing this.',
  },
  'ch24/start/q-limit-fit': {
    compute: 'That row is a rate. This card asks a yes-or-no question about bytes.',
    membw: 'That row is a rate. This card asks a yes-or-no question about bytes.',
    hostlink: 'A link has a rate. This card asks whether bytes fit.',
    gpulink: 'A link has a rate. This card asks whether bytes fit.',
  },
  'ch24/start/q-limit-allreduce': {
    compute: "The arithmetic is one elementwise reduction. The card is about getting each GPU's data to the others.",
    membw: 'Each GPU reads its own memory quickly. The data then has to reach the other GPUs.',
    memcap: 'Nothing here asks whether data fits.',
    hostlink: 'On a node with no direct link the traffic does go over PCIe, which NCCL also supports (p. 564). The row that names a link between GPUs is another one.',
  },
};

const limitQuestion = (qid: string): ChoiceQ => {
  const c = LIMITS.find((x) => x.qid === qid)!;
  return {
    kind: 'choice',
    id: qid,
    prompt: <>Which row of the book's last table limits this? {c.card}</>,
    options: LIMIT_ROWS.map((r) => ({
      label: r.name,
      correct: r.id === c.row,
      why: r.id === c.row ? undefined : NOT_THIS[qid][r.id],
    })),
    explain: <p>{c.why}</p>,
    source: c.source,
  };
};

function LimitsBench() {
  const { progress, answer, recordReview } = useProgress();
  const prior = () =>
    Object.fromEntries(
      LIMITS.map((c) => {
        const v = progress.answers[c.qid]?.value;
        return [c.qid, typeof v === 'number' && LIMIT_ROWS[v] ? LIMIT_ROWS[v].id : ''];
      }),
    ) as Record<string, LimitId | ''>;
  const [picks, setPicks] = useState<Record<string, LimitId | ''>>(prior);
  // The placements as they stood at the last press of Check. Feedback shows only while a row is unchanged since.
  const [checked, setChecked] = useState<Record<string, LimitId | ''> | null>(() =>
    LIMITS.every((c) => progress.answers[c.qid]) ? prior() : null,
  );
  const ready = LIMITS.every((c) => picks[c.qid] !== '');
  const allRight = checked !== null && LIMITS.every((c) => checked[c.qid] === c.row && picks[c.qid] === c.row);
  const check = () => {
    for (const c of LIMITS) {
      const pick = picks[c.qid];
      if (pick === '' || (checked && checked[c.qid] === pick)) continue;
      const correct = pick === c.row;
      if (!progress.answers[c.qid]) recordReview(c.qid, correct); // first answer schedules it
      answer(c.qid, LIMIT_ROWS.findIndex((r) => r.id === pick), correct);
    }
    setChecked(picks);
  };
  return (
    <Bench
      title="What limits what"
      task="Five things you have met. Place each on the row of the table that limits it, then check."
      foot="Each card is limited by a different row. Sources are the earlier chapters named in the feedback; the placement of each card is how those chapters reason, not something chapter 24 says."
    >
      <div className="panel">
        {LIMITS.map((c, i) => {
          const pick = picks[c.qid];
          const fresh = checked !== null && checked[c.qid] === pick && pick !== '';
          const ok = pick === c.row;
          return (
            <div
              key={c.qid}
              data-card={c.qid}
              data-result={fresh ? (ok ? 'right' : 'wrong') : ''}
              style={{ padding: '0.65rem 0', borderTop: i ? '1px solid var(--rule-soft)' : undefined }}
            >
              <div className="row">
                <p style={{ margin: 0, flex: '1 1 18rem', minWidth: 0 }}>{c.card}</p>
                <label className="field" style={{ maxWidth: '100%' }}>
                  <select
                    style={{ maxWidth: '100%' }}
                    aria-label={`Limit for: ${c.card}`}
                    value={pick}
                    onChange={(e) => setPicks({ ...picks, [c.qid]: e.target.value as LimitId | '' })}
                  >
                    <option value="">limited by…</option>
                    {LIMIT_ROWS.map((r) => (
                      <option key={r.id} value={r.id}>
                        {r.name}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              {fresh ? (
                <p style={{ margin: '0.4rem 0 0' }} role="status">
                  {ok ? (
                    <>
                      <span className="chip ok">right</span> {c.why} <span className="muted">Source: {c.source}.</span>
                    </>
                  ) : (
                    <>
                      <span className="chip fail">not {LIMIT_ROWS.find((r) => r.id === pick)!.name}</span>{' '}
                      {NOT_THIS[c.qid][pick as LimitId]}
                    </>
                  )}
                </p>
              ) : null}
            </div>
          );
        })}
      </div>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        <button className="primary" onClick={check} disabled={!ready}>
          Check
        </button>
        <span data-readout="limits" aria-live="polite">
          {checked === null ? (
            <span className="muted">{ready ? 'All five placed.' : 'Place all five first.'}</span>
          ) : allRight ? (
            <span className="chip ok">All five right.</span>
          ) : (
            <>
              {LIMITS.filter((c) => checked[c.qid] === c.row).length} of {LIMITS.length} right at the last check.
            </>
          )}
        </span>
      </div>
      {allRight ? (
        <div className="panel" style={{ marginTop: '0.8rem' }} data-limits-done>
          <p style={{ margin: 0 }}>
            The table's rows are the five limits you already reason with. An inference path on one GPU touches four of
            them: the host link once, when the weights are loaded; capacity when you size the batch; compute in
            prefill; bandwidth for every generated token.
          </p>
        </div>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- 3. check the table
const ROW_NAME: Record<RowId, string> = {
  fp32: 'Compute throughput, FP32',
  tensor: 'Compute throughput, tensor-core figure',
  membw: 'Memory bandwidth',
  memcap: 'Memory capacity',
  pcie: 'Host link (PCIe x16, each way)',
  nvlink: 'GPU-to-GPU link (NVLink)',
};
const ROWS: RowId[] = BOOK.rows.map((r) => r.id);
const lower = (s: string) => (/^[A-Z][a-z]/.test(s) ? s[0].toLowerCase() + s.slice(1) : s);
/** Thresholds: three significant digits, but 137.5 stays 137.5. */
const thrFmt = (v: number) => (v >= 100 ? String(Number(v.toFixed(1))) : fmt(v));
const show = (v: number) => (v >= 1000 ? fmtInt(v) : Number(v.toPrecision(v < 1 ? 4 : 3)).toString());

function Seg<T extends string | number>({ label, value, options, onChange }: { label: string; value: T; options: [T, string][]; onChange: (v: T) => void }) {
  return (
    <label className="field" data-seg={label}>
      {label}
      <span className="seg">
        {options.map(([v, text]) => (
          <button key={String(v)} aria-pressed={value === v} onClick={() => onChange(v)}>
            {text}
          </button>
        ))}
      </span>
    </label>
  );
}

function Recomputed() {
  const [set, setSet] = useState<FigureSet>('book');
  const [o, setO] = useState<Options>(DEFAULTS.book);
  const [sel, setSel] = useState<RowId>('fp32');
  const pick = (s: FigureSet) => {
    setSet(s);
    setO({ ...o, g80Pcie: DEFAULTS[s].g80Pcie });
  };
  const rated = ROWS.filter((r) => r !== 'nvlink').map((r) => ({ r, x: ratio(r, set, o)! }));
  const order = [...rated].sort((a, b) => b.x - a.x);
  const least = order[order.length - 1];
  const next = order[order.length - 2];
  const LOG_MAX = 4; // the axis runs from 1x to 10,000x
  return (
    <div data-explorer style={{ marginTop: '1rem' }}>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <Seg label="Figures" value={set} onChange={pick} options={[['book', 'the book, as printed'], ['vendor', `NVIDIA's documents (${VENDOR.boards.g80}, ${VENDOR.boards.b200})`]]} />
        <Seg label="Tensor-core figure" value={o.tensor} onChange={(v) => setO({ ...o, tensor: v })} options={[['sparse', 'with sparsity'], ['dense', 'dense']]} />
        <Seg label="PCIe generation of the 2007 board" value={o.g80Pcie} onChange={(v) => setO({ ...o, g80Pcie: v })} options={[[1, '1'], [2, '2']]} />
        <Seg label="NVLink counted" value={o.nvlink} onChange={(v) => setO({ ...o, nvlink: v })} options={[['both', 'both directions'], ['eachWay', 'each way']]} />
      </div>
      <div className="panel" style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0, minWidth: '42rem' }} data-recomputed>
          <thead>
            <tr>
              <th>Row (select one)</th>
              <th className="num">2007</th>
              <th className="num">2025</th>
              <th className="num">Growth</th>
              <th className="num" style={{ paddingRight: '1.2rem' }}>Printed</th>
              <th style={{ width: '36%', minWidth: '15rem' }}>
                <div style={{ position: 'relative', height: '1.1em', fontWeight: 450, fontSize: '0.7rem' }} className="muted">
                  {[0, 1, 2, 3, 4].map((e) => (
                    <span key={e} style={{ position: 'absolute', left: `${(e / LOG_MAX) * 100}%`, transform: e === LOG_MAX ? 'translateX(-100%)' : e ? 'translateX(-50%)' : undefined, whiteSpace: 'nowrap' }}>
                      {fmtX(10 ** e)}
                    </span>
                  ))}
                </div>
              </th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((r) => {
              const f = figures(r, set, o);
              const x = ratio(r, set, o);
              const printed = BOOK.rows.find((b) => b.id === r)!.ratio;
              const differs = x != null && fmtX(x) !== printed;
              return (
                <tr key={r} data-row={r} style={sel === r ? { background: 'var(--rule-soft)' } : undefined}>
                  <td>
                    <label className="check" style={{ alignItems: 'baseline' }}>
                      <input type="radio" name="ch24-row" checked={sel === r} onChange={() => setSel(r)} />
                      <span style={sel === r ? { fontWeight: 600 } : undefined}>{ROW_NAME[r]}</span>
                    </label>
                  </td>
                  <td className="num nowrap">{f.from == null ? 'N/A' : `${show(f.from)} ${f.unit}`}</td>
                  <td className="num nowrap">
                    {show(f.to)} {f.unit}
                  </td>
                  <td className="num" data-ratio>
                    <b>{x == null ? '' : fmtX(x)}</b>
                  </td>
                  <td className="num" data-printed style={{ paddingRight: '1.2rem' }}>
                    {differs ? <span className="t-fail">{printed}</span> : <span className="muted">{printed}</span>}
                  </td>
                  <td>
                    {x == null ? (
                      <span className="muted">no 2007 figure to divide by</span>
                    ) : (
                      <div
                        style={{
                          position: 'relative',
                          height: '0.95rem',
                          marginTop: '0.15rem',
                          backgroundImage: 'linear-gradient(90deg, var(--rule) 1px, transparent 1px)',
                          backgroundSize: `${100 / LOG_MAX}% 100%`,
                          borderRight: '1px solid var(--rule)',
                        }}
                      >
                        <div data-bar style={{ width: `${Math.min(100, (Math.log10(x) / LOG_MAX) * 100)}%`, height: '100%', background: 'var(--ink)', borderRadius: '0 2px 2px 0' }} />
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="cols" style={{ marginTop: '0.8rem' }}>
        <div className="panel" data-readout="order" aria-live="polite">
          <p style={{ margin: '0 0 0.4rem' }}>
            <b>Least growth: {lower(ROW_NAME[least.r])}, {fmtX(least.x)}.</b> The next row up is{' '}
            {lower(ROW_NAME[next.r])} at {fmtX(next.x)}, {fmt(next.x / least.x, 2)} times as much.
          </p>
          <p style={{ margin: 0 }} className="muted">
            Bars are on a log scale: each gridline is ten times the one before. NVLink has no 2007 figure; in one
            direction it carries {VENDOR.b200.nvlinkEachWay.value} GB/s, {nvlinkOverHostLink().toFixed(1)} times a Gen5
            x16 host link.
          </p>
        </div>
        <div className="panel" data-readout="built" aria-live="polite">
          <p style={{ margin: '0 0 0.4rem' }}>
            <b>{ROW_NAME[sel]}: how the figures are built</b>
          </p>
          {built(sel, set, o).map((b) => (
            <p key={b.when} style={{ margin: '0 0 0.4rem' }}>
              <b>{b.when}.</b> {b.text} <span className="muted">[facts {b.fact}]</span>
            </p>
          ))}
        </div>
      </div>
    </div>
  );
}

// Cell padding for the printed table, so its four columns fit a phone without scrolling.
const TIGHT: CSSProperties = { padding: '0.4rem 0.25rem' };

function TableAudit() {
  const { progress } = useProgress();
  const committed = typeof progress.answers[P_HOSTLINK]?.value === 'number';
  const cells = useMemo(printedCells, []);
  const byId = useMemo(() => Object.fromEntries(cells.map((c) => [c.id, c])), [cells]);
  const [ticks, setTicks] = useState<string[]>([]);
  // The ticks as they stood at the last press of Check.
  const [checked, setChecked] = useState<string[] | null>(null);
  const toggle = (id: string) => {
    setTicks(ticks.includes(id) ? ticks.filter((t) => t !== id) : [...ticks, id]);
    setChecked(null);
  };
  const wrong = cells.filter((c) => c.wrong);
  const found = checked ? wrong.filter((c) => checked.includes(c.id)).length : 0;
  const others = checked ? checked.filter((id) => !byId[id].wrong).length : 0;
  const listed = checked ? cells.filter((c) => checked.includes(c.id) || c.wrong) : [];

  if (!committed)
    return (
      <Bench title="Check the table" task="Commit your prediction above first.">
        <p style={{ margin: 0 }} data-gate="table">
          The table answers the question you were just asked, so it stays hidden until you have committed a number.
        </p>
      </Bench>
    );

  const cell = (id: string, text: string | null) => {
    if (text == null) return <td className="muted" style={TIGHT}>(the cell above)</td>;
    const c = byId[id];
    if (!c) return <td style={TIGHT}>{text}</td>;
    const ticked = ticks.includes(id);
    const state = !checked ? '' : c.wrong ? 'wrong' : !ticked ? '' : c.stands ? 'stands' : 'ok';
    return (
      <td data-cell={id} data-state={state} style={{ ...TIGHT, background: state === 'wrong' ? 'var(--fail-wash)' : state === 'ok' ? 'var(--ok-wash)' : undefined }}>
        <label className="check" style={{ alignItems: 'baseline', gap: '0.3rem' }}>
          <input type="checkbox" checked={ticked} onChange={() => toggle(id)} aria-label={`${text}: wrong as printed`} />
          <span style={state === 'wrong' ? { color: 'var(--fail)', fontWeight: 600 } : undefined}>{text}</span>
        </label>
      </td>
    );
  };

  return (
    <Bench
      title="Check the table"
      task="Tick every cell you believe is wrong as printed, then check. A calculator is fair."
      foot="The cells are the book's, as printed on p. 578; the layout is ours. The book prints the two compute figures in one cell (“80 TFLOPS / 2200 TFLOPS (FP32 / Tensor Core FP32)”, “232x / 638x”); they are on two lines here so you can tick them separately."
    >
      <div className="panel" style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0, minWidth: '16rem', fontSize: 'clamp(0.7rem, 2.7vw, 0.84rem)' }} data-printed-table>
          <thead>
            <tr>
              <th />
              {BOOK.columns.map((c) => (
                <th key={c} style={TIGHT}>
                  {/* the heading as printed, allowed to break after its slash on a narrow screen */}
                  {c.split('/').map((part, i) => (
                    <span key={i}>
                      {i ? (
                        <>
                          /<wbr />
                        </>
                      ) : null}
                      {part}
                    </span>
                  ))}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {BOOK.rows.map((r) => (
              <tr key={r.id}>
                <td style={TIGHT}>
                  <b>{r.id === 'tensor' ? '' : r.label}</b>
                </td>
                {cell(`${r.id}.g80`, r.g80)}
                {cell(`${r.id}.b200`, r.b200)}
                {cell(`${r.id}.ratio`, r.ratio)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        <button className="primary" onClick={() => setChecked(ticks)}>
          Check
        </button>
        <button className="quiet" onClick={() => { setTicks([]); setChecked(null); }} disabled={!ticks.length}>
          Clear ticks
        </button>
        <span data-readout="audit" aria-live="polite">
          {checked === null ? (
            <span className="muted">
              {ticks.length} of {cells.length} cells ticked.
            </span>
          ) : (
            <>
              <b>Two cells are wrong as printed.</b> You found {found} of {wrong.length}
              {others ? `, and ticked ${others} other ${others === 1 ? 'cell' : 'cells'}` : ''}.
            </>
          )}
        </span>
      </div>
      {checked ? (
        <>
          <div className="panel" style={{ marginTop: '0.8rem' }} data-verdicts>
            <ul style={{ margin: 0, paddingLeft: 0, listStyle: 'none' }}>
              {listed.map((c) => (
                <li key={c.id} data-verdict={c.id} style={{ marginBottom: '0.45rem' }}>
                  {c.wrong ? (
                    <span className="chip fail">{checked.includes(c.id) ? 'wrong as printed' : 'missed'}</span>
                  ) : c.stands ? (
                    <span className="chip" style={{ background: 'var(--rule-soft)' }}>the figure stands</span>
                  ) : (
                    <span className="chip ok">consistent</span>
                  )}{' '}
                  <b>{c.text}</b>: <span className="mono">{c.arithmetic}</span>. {c.verdict}
                </li>
              ))}
            </ul>
            <p className="muted" style={{ margin: '0.3rem 0 0' }}>
              Consistent means the ratio follows from the two figures beside it: {auditPrinted().filter((a) => a.matches).length} of
              the 5 printed ratios do. Whether each figure is the right one for the hardware is a separate question,
              and the controls below take it up.
            </p>
          </div>
          <Recomputed />
        </>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- 4. where the threshold went
function measured(reports: Reports): Device[] {
  return realBackends(reports).flatMap((b) => {
    const r = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
    const peak = num(r?.compute_gflops);
    const bw = num(r?.bandwidth_gb_per_s);
    if (!peak || !bw) return [];
    return [
      {
        key: b,
        label: `Yours: ${String(reports[b].device.name ?? b)} (${b}), measured by lab 5.1`,
        peak,
        bw,
        note: `Measured by lab 5.1 on ${reports[b].generated.slice(0, 10)} with two simple probe kernels. It is not a vendor peak.`,
      },
    ];
  });
}

function Strip({ thr, tile }: { thr: number; tile: number }) {
  const W = 760;
  const H = 222;
  const pad = { l: 18, r: 18 };
  const axisY = 150;
  const xMin = 0.05;
  const xMax = 2000;
  const X = (v: number) =>
    pad.l + ((Math.log10(Math.min(Math.max(v, xMin), xMax)) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin))) * (W - pad.l - pad.r);
  const ms = marks(tile);
  // Labels sit on three levels so neighbours do not collide.
  const level = [2, 1, 0, 2, 1, 0];
  const tx = X(thr);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Kernels on a FLOP per byte axis with the device threshold" style={{ width: '100%', minWidth: 560, display: 'block' }}>
      <rect x={pad.l} y={14} width={W - pad.l - pad.r} height={axisY - 14} fill="var(--page)" />
      <rect x={pad.l} y={14} width={Math.max(0, tx - pad.l)} height={axisY - 14} fill="var(--global-wash)" />
      <rect x={tx} y={14} width={Math.max(0, W - pad.r - tx)} height={axisY - 14} fill="var(--reg-wash)" />
      <text x={pad.l + 6} y={27} fontSize="11" style={{ fill: 'var(--global)' }}>memory-bound</text>
      <text x={W - pad.r - 6} y={27} fontSize="11" textAnchor="end" style={{ fill: 'var(--reg)' }}>compute-bound</text>
      <line x1={pad.l} x2={W - pad.r} y1={axisY} y2={axisY} stroke="var(--ink)" strokeWidth="1.4" />
      {[0.1, 1, 10, 100, 1000].map((v) => (
        <g key={v}>
          <line x1={X(v)} x2={X(v)} y1={axisY} y2={axisY + 5} stroke="var(--ink)" />
          <text x={X(v)} y={axisY + 18} fontSize="11" textAnchor="middle">{v}</text>
        </g>
      ))}
      <text x={W / 2} y={H - 6} fontSize="11.5" textAnchor="middle">operations per byte of global-memory traffic (FLOP/B), log scale</text>
      {ms.map((m, i) => {
        const cx = X(m.ratio);
        const ly = 50 + level[i] * 30;
        const tone = regime(m.ratio, thr) === 'memory-bound' ? 'var(--global)' : 'var(--reg)';
        const anchor = cx < 110 ? 'start' : cx > W - 110 ? 'end' : 'middle';
        return (
          <g key={m.id} data-mark={m.id}>
            <line x1={cx} x2={cx} y1={ly + 5} y2={axisY} stroke={tone} strokeWidth={m.tiled ? 1.8 : 1} />
            <circle cx={cx} cy={axisY} r={m.tiled ? 6 : 4.5} fill={m.tiled ? tone : 'var(--page)'} stroke={tone} strokeWidth="1.8" />
            <text x={anchor === 'start' ? cx - 4 : anchor === 'end' ? cx + 4 : cx} y={ly} fontSize="11" textAnchor={anchor} fontWeight={m.tiled ? 650 : 400} style={{ paintOrder: 'stroke', stroke: 'var(--page)', strokeWidth: 4, strokeLinejoin: 'round' }}>
              {m.label}
            </text>
          </g>
        );
      })}
      <line x1={tx} x2={tx} y1={8} y2={axisY + 30} stroke="var(--ink)" strokeWidth="2" strokeDasharray="5 3" />
      <text x={tx + (tx > W - 150 ? -6 : 6)} y={axisY + 42} fontSize="11.5" fontWeight="650" textAnchor={tx > W - 150 ? 'end' : 'start'}>
        threshold {thrFmt(thr)}
      </text>
    </svg>
  );
}

function ThresholdBench() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const committed = typeof progress.answers[P_KNEE]?.value === 'number';
  const mine = useMemo(() => measured(reports), [reports]);
  const [key, setKey] = useState('book2007');
  const [dense, setDense] = useState(false);
  const [tile, setTile] = useState(15);
  if (!committed)
    return (
      <Bench title="Where the threshold went" task="Commit your prediction above first.">
        <p style={{ margin: 0 }} data-gate="knee">
          The strip shows the number you were just asked for, so it stays hidden until you have committed one.
        </p>
      </Bench>
    );
  const offered = [...DEVICES.filter((d) => d.key !== 'book2025dense'), ...mine];
  const dev = key === 'book2025tc' && dense ? DEVICES.find((d) => d.key === 'book2025dense')! : (offered.find((d) => d.key === key) ?? DEVICES[0]);
  const thr = threshold(dev.peak, dev.bw);
  const base = DEVICES[0];
  const baseThr = threshold(base.peak, base.bw);
  const need = tileFor(thr);
  const ms = marks(tile);
  return (
    <Bench
      title="Where the threshold went"
      task="Start on the 2007 column and find the first T that crosses. Then change the device."
      foot={<>Log scale. Marks are kernel ratios from earlier chapters; the filled one moves with T. Left of the dashed line a kernel is memory-bound on the chosen figures. {dev.note}</>}
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          Device figures
          <select value={key} onChange={(e) => setKey(e.target.value)} aria-label="Device figures" style={{ maxWidth: '100%' }}>
            {offered.map((d) => (
              <option key={d.key} value={d.key}>
                {d.label}
              </option>
            ))}
          </select>
        </label>
        {key === 'book2025tc' ? (
          <label className="check">
            <input type="checkbox" checked={dense} onChange={(e) => setDense(e.target.checked)} />
            use NVIDIA's dense TF32 figure
          </label>
        ) : null}
        <label className="field">
          Tile width T = {tile}
          <input type="range" min="1" max="128" value={tile} onChange={(e) => setTile(+e.target.value)} aria-label="Tile width T" />
        </label>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <Strip thr={thr} tile={tile} />
      </div>
      <div className="cols" style={{ marginTop: '0.8rem' }}>
        <div className="panel" data-readout="knee" aria-live="polite">
          <p style={{ fontSize: '1.4rem', margin: '0 0 0.2rem', fontWeight: 600 }}>threshold {thrFmt(thr)} FLOP/B</p>
          <p style={{ margin: '0 0 0.5rem' }} className="muted">
            {fmtInt(dev.peak)} GFLOP/s over {show(dev.bw)} GB/s
            {dev.key.startsWith('book') && dev.key !== 'book2007' ? <>, {fmt(thr / baseThr, 2)} times the 2007 column's {thrFmt(baseThr)}</> : null}.
          </p>
          <p style={{ margin: '0 0 0.4rem' }}>
            Smallest T with 0.25 × T at or above it: <b data-need>{fmtInt(need)}</b>
            {need > 128 ? ', far off this slider' : ''}.
          </p>
          <p style={{ margin: 0 }}>
            At T = {tile} the tiled kernel reaches {fmt(0.25 * tile)} FLOP/B:{' '}
            {regime(0.25 * tile, thr) === 'memory-bound' ? <span className="chip global">memory-bound</span> : <span className="chip reg">compute-bound</span>}
          </p>
          {mine.length === 0 ? (
            <p className="muted" style={{ margin: '0.6rem 0 0' }}>
              No device of yours is offered: lab 5.1 has not been benchmarked on this machine (<a href={href('ch05', 'labs')}>chapter 5 labs</a>).
            </p>
          ) : null}
        </div>
        <div className="panel" style={{ overflowX: 'auto' }}>
          <table className="data" style={{ margin: 0 }} data-marks>
            <thead>
              <tr>
                <th>Kernel</th>
                <th className="num">FLOP/B</th>
                <th>On these figures</th>
                <th>From</th>
              </tr>
            </thead>
            <tbody>
              {[...ms].sort((a, b) => a.ratio - b.ratio).map((m) => (
                <tr key={m.id} data-mark-row={m.id}>
                  <td>{m.label}</td>
                  <td className="num">{fmt(m.ratio)}</td>
                  <td>{regime(m.ratio, thr) === 'memory-bound' ? <span className="chip global">memory-bound</span> : <span className="chip reg">compute-bound</span>}</td>
                  <td className="muted">{m.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const BOOK_2007 = DEVICES.find((d) => d.key === 'book2007')!;
const BOOK_2025 = DEVICES.find((d) => d.key === 'book2025')!;
const BOOK_TC = DEVICES.find((d) => d.key === 'book2025tc')!;
const THR_2007 = threshold(BOOK_2007.peak, BOOK_2007.bw);
const THR_2025 = threshold(BOOK_2025.peak, BOOK_2025.bw);
const THR_TC = threshold(BOOK_TC.peak, BOOK_TC.bw);
const TENSOR_AUDIT = auditPrinted().find((a) => a.row === 'tensor')!;

export const questions240: Record<string, Question> = {
  'ch24/start/q-compose': {
    kind: 'choice',
    id: 'ch24/start/q-compose',
    prompt: (
      <>
        You have a tested tiled GEMM and want top-p sampling to run on the device. Going by the book's three steps,
        where is the material?
      </>
    ),
    options: [
      {
        label: 'In step three, because sampling is an application, like the chapter on large language models.',
        why: 'The application chapter, 20, stops at attention. It has no sampler.',
      },
      {
        label:
          'In step two, spread over four pattern chapters. The book never assembles them into a sampler, so the assembly is a module of ours.',
        correct: true,
      },
      {
        label: 'In step one, because sampling is an elementwise operation on the logits.',
        why: 'The top-p cut-off depends on a running sum over the whole row, so no thread can decide alone.',
      },
      {
        label: 'Nowhere. Sampling needs techniques the book does not teach.',
        why: 'Every building block is a pattern chapter. Only the assembly is missing.',
      },
    ],
    explain: (
      <p>
        The normaliser and the maximum are reductions (chapter 10), the cumulative probability is a scan (chapter 11),
        dropping the candidates past the cut-off is a filter (chapter 12), and putting them in order is a sort (chapter
        14). All four are step two. Module X1 puts them together, on top of the softmax you build in chapter 10.
      </p>
    ),
    source: 'ch. 24 §24.1, pp. 577–578, for the steps; the course plan for what the book lacks',
  },
  ...Object.fromEntries(LIMITS.map((c) => [c.qid, limitQuestion(c.qid)])),
  'ch24/start/q-638': {
    kind: 'numeric',
    id: 'ch24/start/q-638',
    prompt: (
      <>
        The table prints 0.345 TFLOPS for 2007, 2200 TFLOPS for the 2025 tensor-core figure, and a ratio of 638x. What
        2025 figure would make 638x correct?
      </>
    ),
    answer: TENSOR_AUDIT.wouldMatch!,
    tol: 0.01,
    unit: 'TFLOPS',
    explain: (
      <p>
        638 × 0.345 = 220. So either the 2200 or the 638 is off by ten. NVIDIA prints 2.2 petaFLOPS for this row (TF32
        with sparsity) and no source we found prints 220, so the ratio is the wrong cell. It should read about{' '}
        {fmtX(TENSOR_AUDIT.recomputed)}.
      </p>
    ),
    source: 'ch. 24 §24.2, Fig. 24.1, p. 578; notes/today/ch24-facts.md, A9',
  },
  'ch24/start/q-tile': {
    kind: 'numeric',
    id: 'ch24/start/q-tile',
    prompt: (
      <>
        A square-tiled matmul with one thread per output element reaches 0.25 × T FLOP/B. How wide must T be to reach
        the threshold computed from the book's 2025 tensor-core figure, {fmt(THR_TC)} FLOP/B?
      </>
    ),
    answer: tileFor(THR_TC),
    tol: 0.01,
    explain: (
      <>
        <p>
          {fmt(THR_TC)} / 0.25 = {fmtInt(tileFor(THR_TC))}. Two T × T tiles of floats would need 8 × T × T bytes of
          shared memory, about 9.7 MB per block. Compare that with the limit <code>python3 labs/pmpp.py doctor</code>{' '}
          prints for your device (<code>sharedMemPerBlock</code> on CUDA and HIP).
        </p>
        <p>
          The reuse has to come from somewhere else: larger output tiles held in registers (chapter 15), batching, or
          fewer bytes per value. On the 2007 figures the answer was {tileFor(THR_2007)}.
        </p>
      </>
    ),
    source: 'our analysis with the §5.1 method (ch. 5, pp. 94–97) on the figures of Fig. 24.1, p. 578',
  },
  'ch24/start/q-norow': {
    kind: 'choice',
    id: 'ch24/start/q-norow',
    prompt: (
      <>
        An inference path on one GPU touches four rows of the book's table. Which of the four has no counterpart on
        Apple silicon, and what cost takes its place?
      </>
    ),
    options: [
      {
        label: 'Memory bandwidth. Unified memory is fast enough that the row no longer limits a kernel.',
        why: 'Apple publishes a chip-level memory bandwidth figure. The row exists.',
      },
      {
        label: 'Memory capacity. With unified memory the device has no memory of its own to run out of.',
        why: 'Unified memory still has a size, and the weights and KV cache still have to fit in it.',
      },
      {
        label:
          'The host-to-device link. A shared buffer is visible to both sides with no copy; what you avoid instead is waiting for a command buffer to complete before the CPU reads results.',
        correct: true,
      },
      {
        label: 'Compute throughput. Apple publishes no FLOPS figure.',
        why: 'The published number is missing, but the limit on a kernel is still there.',
      },
    ],
    explain: (
      <p>
        On Apple silicon the CPU and the GPU share system memory, so there is no bus to cross and no figure to put in
        that row. The discipline the row stands for changes shape: keep results on the GPU side across kernels, because
        each time the CPU needs them it has to wait for the work to finish. The contrast is ours, not Apple's and not
        the book's.
      </p>
    ),
    source: 'notes/today/ch05-facts.md, C6; notes/today/ch24-facts.md, C2',
  },
};

// ---------------------------------------------------------------- the page
const BACKEND_ROWS: { role: string; cuda: ReactNode; hip: ReactNode; metal: ReactNode; facts: string }[] = [
  {
    role: 'Matrix-multiply hardware',
    cuda: 'Tensor Cores',
    hip: 'Matrix cores, formally MFMA units; documented for CDNA parts only',
    metal: 'None named in Metal documentation',
    facts: 'A9, B1, C1',
  },
  {
    role: 'How kernel code reaches it',
    cuda: (
      <>
        CUDA Tile (<code>cuda::tiles</code> in <code>cuda_tile.h</code>, Toolkit 13.3 onward), or libraries
      </>
    ),
    hip: 'Libraries, compiler intrinsics or inline assembly; no tile-level model found',
    metal: (
      <>
        MSL library calls: <code>mpp::tensor_ops::matmul2d</code>, Apple's stated preference since Metal 4, or{' '}
        <code>&lt;metal_simdgroup_matrix&gt;</code>
      </>
    ),
    facts: 'A19, B1, C1',
  },
  { role: 'Host-to-device link', cuda: 'PCIe', hip: 'PCIe', metal: 'None: memory is unified', facts: 'A12, B4, C2' },
  { role: 'GPU-to-GPU link', cuda: 'NVLink', hip: 'Infinity Fabric links', metal: 'None documented for Apple silicon', facts: 'A13, B2, C3' },
  {
    role: 'Collective library',
    cuda: 'NCCL',
    hip: (
      <>
        RCCL, with the same <code>ncclXxx</code> function names
      </>
    ),
    metal: (
      <>
        None in Metal; MLX has <code>mx.distributed</code>
      </>
    ),
    facts: 'A16, B3, C4',
  },
];

export function S240() {
  return (
    <>
      <p>
        Chapter 24 is four pages with no code and no exercises: the book's goals revisited, then the authors' outlook.
        This page keeps three things: the three steps as a statement of what you can now do, the chapter's one table
        checked line by line, and a reading of that table for kernel work.
      </p>
      <Source pages="577" />

      <h2>What you can now do</h2>
      <p>
        The book promised two things: computational thinking, and a systematic way to attack whatever limits
        performance. Step one, chapters 2 to 6: write a kernel, map threads to data, reason about warps, occupancy,
        memory tiers and coalescing, and end with the optimisation checklist (§6.8, pp. 147–153). Step two, chapters 7
        to 15: the patterns, each a vehicle for those optimisations. Step three, chapters 16 to 23: alternative
        decompositions and their data structures, with chapter 22 as the recap of the thinking. With this, the authors
        claim (p. 578), performance work is a reasoning process. The grouping is loose: chapters 22 and 23 count as
        “advanced patterns and applications”.
      </p>
      <Source section="24.1" pages="577–578" />
      <StepsMap />
      <Ask q={questions240['ch24/start/q-compose']} />

      <h2>What limits what</h2>
      <p>
        The book's last table has five rows, each a limit you already reason with. Before reading it, place five
        familiar things on the row that limits them.
      </p>
      <Aside>
        <p>
          From ch. 2 §2.4 (pp. 28–32), ch. 5 §5.1 (pp. 93–97), ch. 20 §20.4 and §20.6 (pp. 488–492, 504–508) and ch. 23
          §23.4 (pp. 563–568).
        </p>
      </Aside>
      <LimitsBench />

      <h2>The book's one table</h2>
      <p>
        Section 24.2 opens with a table that sets the first CUDA GPU beside a 2025 one, with a ratio column. Before you see it, commit to a number for one of its rows.
      </p>
      <Source section="24.2" pages="578" figs="Fig. 24.1" />
      <Predict
        id={P_HOSTLINK}
        unit="x"
        prompt={
          <>
            From 2007 to 2025 the book's table has compute throughput up 232x, memory bandwidth up 93x and memory
            capacity up 128x. By what factor does it say the host link (PCIe x16, each way) grew?
          </>
        }
      >
        {(v) => (
          <p>
            <b>8x</b> as printed: 8 GB/s to 64 GB/s.{' '}
            {Math.abs(v - 8) <= 1
              ? 'That is what the table prints.'
              : v >= 50
                ? 'A guess near 100 assumes the link kept pace with the device. It did not.'
                : v >= 12 && v <= 20
                  ? "That is close to what NVIDIA's own documents give for the 2007 board."
                  : v < 8
                    ? 'You were lower than the table, and right about the direction: the link fell behind.'
                    : 'The link fell further behind the device than you expected.'}{' '}
            On NVIDIA's figures for the 2007 board the factor is about 16x; you will get to that below.
          </p>
        )}
      </Predict>
      <TableAudit />
      <Ask q={questions240['ch24/start/q-638']} />
      <p>
        Two cells are wrong as printed. 2200 / 0.345 is 6,377, not 638, and the 2,200 figure is NVIDIA's own, so the
        ratio is the wrong cell. “1,800 TB/s” is 1,800 GB/s, and it counts both directions together while the PCIe row
        counts each way.
      </p>
      <Aside>
        <p>
          Facts items A9 and A13 for these two; A9, A3 and A4, and A6 for the three that follow
          (<code>notes/today/ch24-facts.md</code>).
        </p>
      </Aside>
      <p>
        Three more corrections come from NVIDIA's documents. The row labelled “Tensor Core FP32” is TF32 on Tensor
        Cores counted with sparsity; dense is half. The G80 column takes its bandwidth from one board and its capacity
        from another. The 1.5 GB board is PCI Express generation 1 in NVIDIA's board specification, so the host link
        grew about 16x on those figures, not 8x.
      </p>
      <Today checked="2026-11-20 (ch24-facts A2, A8, A10, A11)">
        <p>
          Every figure is a published peak, not a measurement: 0.345 TFLOPS is 128 multiply-adds per clock × 2 × 1.35
          GHz. For shipped HGX B200 boards NVIDIA's documents print 75 TFLOPS FP32 and 180 GB, and “up to 8 TB/s”.
        </p>
      </Today>
      <p>
        One point survives every correction: everything on the device grew by two orders of magnitude or more, the
        host link by one. Our reading (the book only prints the numbers): keep weights, the KV cache and sampling on
        the device.
      </p>
      <Today checked="2026-11-20 (ch24-facts A1, A7, A15)">
        <p>
          The G80 reached shops in November 2006; the first CUDA Toolkit is from June 2007. The B200 was announced in
          March 2024, and NVIDIA's pages already list newer parts, so the 2025 column is superseded.
        </p>
      </Today>

      <h2>Reading the table as a kernel writer</h2>
      <p>
        The book stops at the ratios. Section 5.1 gives the top two rows a better use: a device's threshold is its
        peak throughput over its peak bandwidth, and a kernel whose FLOP/B is below it is memory-bound.
      </p>
      <Beyond>
        <p>
          The method is the book's (§5.1, pp. 94–97). Applying it to Fig. 24.1 is ours; chapter 24 does not do it.
        </p>
      </Beyond>
      <p>
        The strip puts kernels from earlier chapters on one FLOP/B axis. Choose the device figures and the threshold
        moves; everything left of it is memory-bound on those figures.
      </p>
      <Simplification>
        <p>
          Peaks are vendor figures of different kinds, and the 2007 column mixes two boards. The tensor-core peak is
          TF32 arithmetic with sparsity and applies only to matmul-shaped work.
        </p>
        <p>
          The model counts the traffic a kernel asks for and ignores caches. “Yours” is measured by lab 5.1's probe
          kernels and is not a vendor peak.
        </p>
      </Simplification>
      <Predict
        id={P_KNEE}
        unit="FLOP/B"
        prompt={
          <>
            By the book's 2007 figures the threshold of section 5.1 is 0.345 TFLOPS over 86.4 GB/s, about{' '}
            {THR_2007.toFixed(1)} FLOP/B. FP32 throughput then grew 232x and bandwidth 93x. What is the threshold in
            the 2025 column?
          </>
        }
      >
        {(v) => (
          <p>
            80,000 GFLOP/s / 8,000 GB/s = <b>{fmt(THR_2025)} FLOP/B</b>, {fmt(THR_2025 / THR_2007, 2)} times the 2007
            value. Against the tensor-core figure it is {fmt(THR_TC)}.{' '}
            {Math.abs(v - THR_2025) <= 0.5
              ? 'That is what you had.'
              : v < THR_2007
                ? 'A guess below 4 has the quotient upside down: compute grew faster than bandwidth, so the threshold rose.'
                : 'The shortcut is 4.0 × 232 / 93.'}
          </p>
        )}
      </Predict>
      <ThresholdBench />
      <p>
        Compute grew faster than bandwidth, so the threshold rose. A 16-wide tile reached it on the 2007 figures.
        Against the tensor-core figure neither chapter 5's 32-wide tile (8 FLOP/B) nor chapter 15's 128 × 128 output
        tile (32 FLOP/B, §15.2, pp. 350–352) does.
      </p>
      <Ask q={questions240['ch24/start/q-tile']} />
      <p>
        Batch-1 generation, about 1 FLOP/B in the book's 16-bit example (§20.6, p. 505), is memory-bound in every
        column, and its ratio is set by bytes per weight and by batching. That is why the work left is chapter 15,
        batching (chapters 20 and 22) and quantised kernels (module X2). The larger compute number is the mode behind a
        flag you may already have set, <code>torch.backends.cuda.matmul.allow_tf32</code>.
      </p>
      <Today checked="2026-11-20 against the PyTorch 2.15 docs (ch24-facts A9, D4)">
        <p>
          PyTorch leaves that flag off for matmul by default. TF32 rounds inputs to 10 bits of mantissa and accumulates
          in FP32.
        </p>
      </Today>

      <h2>The same rows on AMD and Apple</h2>
      <p>
        The table's vocabulary is NVIDIA's. Here is what each role is called, or whether it exists, on the two other
        backends this course builds for.
      </p>
      <Beyond>
        <p>
          Chapter 24 names no other vendor. Every cell is from <code>notes/today/ch24-facts.md</code>; the last column
          gives the items.
        </p>
      </Beyond>
      <table className="data wide" data-backends>
        <thead>
          <tr>
            <th>Role</th>
            <th>CUDA (NVIDIA)</th>
            <th>HIP (AMD)</th>
            <th>Metal (Apple)</th>
            <th>Facts</th>
          </tr>
        </thead>
        <tbody>
          {BACKEND_ROWS.map((r) => (
            <tr key={r.role}>
              <td>
                <b>{r.role}</b>
              </td>
              <td>{r.cuda}</td>
              <td>{r.hip}</td>
              <td>{r.metal}</td>
              <td className="muted nowrap">{r.facts}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>
        TF32 is not portable: one current AMD part offers it through software emulation, and it is absent from the
        type table of Metal's <code>matmul2d</code>. Apple publishes no FLOPS figure, so no Metal column of numbers can
        be built.
      </p>
      <Today checked="2026-11-20 against AMD's HIP documentation and product page and the MSL specification 4.1 (ch24-facts B1, C1, C3, C5)">
        <p>
          “No tile-level model found” for HIP and “none documented” for Apple mean we did not find one in the pages
          read. Neither vendor says there is none.
        </p>
      </Today>
      <Ask q={questions240['ch24/start/q-norow']} />

      <h2>The rest of the outlook</h2>
      <p>
        The last two pages are the authors' outlook, and all of it is theirs. Jobs now span many GPUs, driven by LLM
        training, with NCCL as the library named. Innovation is now led by computing assisted by instruments, no longer
        the reverse; their examples are driving, medicine and lithography. Parallel access to storage is the next
        opening they see.
      </p>
      <Today checked="2026-11-20 (ch24-facts A16, A17, A18)">
        <p>
          NCCL's public repository dates from 2015, so “new” is relative. The statements about job sizes and about
          lithography are the authors' judgement and only partly checkable against vendor pages.
        </p>
      </Today>
      <p>
        Left in the book: the three examples in full and the fields the authors expect to change next (pp. 579–580).{' '}
        <a href={href('ch24', 'wrap')}>Summary and where next</a> collects what to carry forward.
      </p>
      <Source section="24.2" pages="579–580" />
    </>
  );
}
