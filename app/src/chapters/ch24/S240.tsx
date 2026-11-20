import { useEffect, useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Ask, Predict, type ChoiceQ, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { fmt, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { PLAN, useAllChapters } from '../registry';
import { GOALS, STEPS, goal, lanes, rollup, stepOf, stepsOf, type GoalId, type Step } from '../../sim/booksteps';
import {
  BOOK,
  CORRECTIONS,
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
  type FigureSet,
  type LimitId,
  type Options,
  type RowId,
} from '../../sim/hwtable';

const P_HOSTLINK = 'ch24/start/p-hostlink';
const P_KNEE = 'ch24/start/p-knee';

/** True below 600 px of viewport width, where the two wide widgets lay themselves out again. */
const NARROW = '(max-width: 600px)';
function useNarrow() {
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && window.matchMedia(NARROW).matches);
  useEffect(() => {
    const m = window.matchMedia(NARROW);
    const on = () => setNarrow(m.matches);
    on();
    m.addEventListener('change', on);
    return () => m.removeEventListener('change', on);
  }, []);
  return narrow;
}

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
    hostlink: 'These GPUs are joined to each other directly, and the book calls NCCL topology-aware (p. 564), so their traffic need not cross the link to the host. Without NVLink it would go over PCIe.',
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
      foot="Each card is limited by a different row. Each is labelled with the chapters it draws on, and the feedback gives sections and pages; the placement of each card is how those chapters reason, not something chapter 24 says."
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
              <div className="muted" style={{ fontSize: '0.74rem', fontWeight: 600, marginBottom: '0.2rem' }} data-from>
                {c.from}
              </div>
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

// A printed ratio its own printed figures contradict (only 638x): the one that earns the failure colour.
const CONTRADICTED = new Set(auditPrinted().filter((a) => !a.matches).map((a) => a.row));

function Recomputed() {
  const narrow = useNarrow();
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
        {/* Below 600 px the two figure columns move under the row name, so growth, printed ratio and bar stay in view. */}
        <table className="data" style={{ margin: 0, minWidth: narrow ? 0 : '42rem', fontSize: narrow ? '0.8rem' : undefined }} data-recomputed>
          <thead>
            <tr>
              <th>Row (select one)</th>
              {narrow ? null : (
                <>
                  <th className="num">2007</th>
                  <th className="num">2025</th>
                </>
              )}
              <th className="num">Growth</th>
              <th className="num" style={{ paddingRight: narrow ? undefined : '1.2rem' }}>Printed</th>
              <th style={{ width: narrow ? '22%' : '36%', minWidth: narrow ? '3.5rem' : '15rem' }}>
                {narrow ? (
                  <span className="muted" style={{ fontWeight: 450, fontSize: '0.7rem' }}>
                    log, {fmtX(1)} to {fmtX(10 ** LOG_MAX)}
                  </span>
                ) : (
                  <div style={{ position: 'relative', height: '1.1em', fontWeight: 450, fontSize: '0.7rem' }} className="muted">
                    {[0, 1, 2, 3, 4].map((e) => (
                      <span key={e} style={{ position: 'absolute', left: `${(e / LOG_MAX) * 100}%`, transform: e === LOG_MAX ? 'translateX(-100%)' : e ? 'translateX(-50%)' : undefined, whiteSpace: 'nowrap' }}>
                        {fmtX(10 ** e)}
                      </span>
                    ))}
                  </div>
                )}
              </th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((r) => {
              const f = figures(r, set, o);
              const x = ratio(r, set, o);
              const printed = BOOK.rows.find((b) => b.id === r)!.ratio;
              const differs = x != null && fmtX(x) !== printed;
              const fromText = f.from == null ? 'N/A' : `${show(f.from)} ${f.unit}`;
              const toText = `${show(f.to)} ${f.unit}`;
              return (
                <tr key={r} data-row={r} style={sel === r ? { background: 'var(--rule-soft)' } : undefined}>
                  <td>
                    <label className="check" style={{ alignItems: 'baseline' }}>
                      <input type="radio" name="ch24-row" checked={sel === r} onChange={() => setSel(r)} />
                      <span style={sel === r ? { fontWeight: 600 } : undefined}>{ROW_NAME[r]}</span>
                    </label>
                    {narrow ? (
                      <div className="muted" style={{ fontSize: '0.74rem' }}>
                        {fromText} to <span data-to>{toText}</span>
                      </div>
                    ) : null}
                  </td>
                  {narrow ? null : (
                    <>
                      <td className="num nowrap">{fromText}</td>
                      <td className="num nowrap" data-to>
                        {toText}
                      </td>
                    </>
                  )}
                  <td className="num" data-ratio>
                    <b>{x == null ? '' : fmtX(x)}</b>
                  </td>
                  <td
                    className="num nowrap"
                    data-printed
                    data-differs={!differs ? undefined : CONTRADICTED.has(r) ? 'contradicted' : 'figures'}
                    style={{ paddingRight: narrow ? undefined : '1.2rem' }}
                  >
                    {!differs ? (
                      <span className="muted">{printed}</span>
                    ) : CONTRADICTED.has(r) ? (
                      <span className="t-fail" title="Its own printed figures do not give this ratio">
                        {printed}
                      </span>
                    ) : (
                      <span title="Follows from the printed figures, not from the figures selected">
                        <span className="muted">≠ </span>
                        {printed}
                      </span>
                    )}
                  </td>
                  <td>
                    {x == null ? (
                      <span className="muted">{narrow ? 'no 2007 figure' : 'no 2007 figure to divide by'}</span>
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
          <p style={{ margin: '0 0 0.4rem' }} className="muted">
            Bars are on a log scale: each gridline is ten times the one before. NVLink has no 2007 figure; in one
            direction it carries {VENDOR.b200.nvlinkEachWay.value} GB/s, {nvlinkOverHostLink().toFixed(1)} times a Gen5
            x16 host link.
          </p>
          <p style={{ margin: 0 }} className="muted">
            “≠” marks a printed ratio that follows from the book's figures but not from the ones selected. Red marks
            the one its own printed figures contradict.
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

const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;

function TableAudit({ onCheck }: { onCheck: () => void }) {
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
  const wrong = cells.filter((c) => c.kind === 'wrong');
  const found = checked ? wrong.filter((c) => checked.includes(c.id)).length : 0;
  const others = checked ? checked.filter((id) => byId[id].kind !== 'wrong') : [];
  const othersDiffer = others.filter((id) => byId[id].kind === 'differs').length;
  const othersHold = others.length - othersDiffer;
  const listed = checked ? cells.filter((c) => checked.includes(c.id) || c.kind === 'wrong') : [];

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
    // Red: the table contradicts itself. Green: its arithmetic holds and nothing corrects it.
    // Neutral: its arithmetic holds, but NVIDIA's documents give another figure or label.
    const state = !checked ? '' : c.kind === 'wrong' ? 'wrong' : !ticked ? '' : c.kind;
    const wash = { wrong: 'var(--fail-wash)', consistent: 'var(--ok-wash)', differs: 'var(--rule-soft)', '': undefined }[state];
    return (
      <td data-cell={id} data-state={state} style={{ ...TIGHT, background: wash }}>
        <label className="check" style={{ alignItems: 'baseline', gap: '0.3rem' }}>
          <input type="checkbox" checked={ticked} onChange={() => toggle(id)} aria-label={`${text}: the table contradicts itself here`} />
          <span style={state === 'wrong' ? { color: 'var(--fail)', fontWeight: 600 } : undefined}>{text}</span>
        </label>
      </td>
    );
  };

  return (
    <Bench
      title="Check the table"
      task="Tick every cell the table itself contradicts: a ratio its two figures do not give, or a unit that cannot be meant. Then check. A calculator is fair."
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
        <button
          className="primary"
          onClick={() => {
            setChecked(ticks);
            onCheck();
          }}
        >
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
              <b>The table contradicts itself in {wrong.length === 2 ? 'two' : wrong.length} cells.</b> You found {found} of{' '}
              {wrong.length}
              {others.length
                ? `, and ticked ${plural(others.length, 'other cell', 'other cells')}: ${[
                    othersDiffer ? `${othersDiffer} that NVIDIA's documents correct` : '',
                    othersHold ? `${othersHold} that ${othersHold === 1 ? 'checks' : 'check'} out` : '',
                  ]
                    .filter(Boolean)
                    .join(' and ')}`
                : ''}
              .
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
                  {c.kind === 'wrong' ? (
                    <span className="chip fail">{checked.includes(c.id) ? 'contradicted' : 'missed'}</span>
                  ) : c.kind === 'differs' ? (
                    <span className="chip" style={{ background: 'var(--rule-soft)' }}>
                      NVIDIA's documents differ
                    </span>
                  ) : (
                    <span className="chip ok">consistent</span>
                  )}{' '}
                  <b>{c.text}</b>: <span className="mono">{c.arithmetic}</span>. {c.verdict}
                  {c.fact ? <span className="muted"> [facts {c.fact}]</span> : null}
                </li>
              ))}
            </ul>
            <p className="muted" style={{ margin: '0.3rem 0 0' }}>
              Consistent means the table's arithmetic holds: {auditPrinted().filter((a) => a.matches).length} of the 5
              printed ratios follow from the two figures beside them. Whether each figure is the right one for the
              hardware is a separate question: NVIDIA's documents correct {Object.keys(CORRECTIONS).length} cells that
              are not contradictions, one of them only in its label. The controls below recompute every ratio on
              NVIDIA's figures.
            </p>
          </div>
          <Recomputed />
        </>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- 4. where the threshold went
function Strip({ thr, tile, narrow }: { thr: number; tile: number; narrow: boolean }) {
  // Below 600 px the strip is drawn on a narrower canvas with larger type, one label per
  // level, so it scales to the box instead of scrolling inside it.
  const W = narrow ? 400 : 760;
  const fs = narrow ? 13 : 11;
  const pad = { l: 18, r: 18 };
  const level = narrow ? [0, 1, 2, 3, 4, 5] : [2, 1, 0, 2, 1, 0]; // label rows, so neighbours do not collide
  const step = narrow ? 20 : 30;
  const top = narrow ? 48 : 50;
  const axisY = narrow ? top + 5 * step + 24 : 150;
  const H = axisY + 72;
  const xMin = 0.05;
  const xMax = 2000;
  const X = (v: number) =>
    pad.l + ((Math.log10(Math.min(Math.max(v, xMin), xMax)) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin))) * (W - pad.l - pad.r);
  const ms = marks(tile);
  const tx = X(thr);
  const edge = narrow ? 90 : 110;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Kernels on a FLOP per byte axis with the device threshold" style={{ width: '100%', display: 'block' }} data-strip>
      <rect x={pad.l} y={14} width={W - pad.l - pad.r} height={axisY - 14} fill="var(--page)" />
      <rect x={pad.l} y={14} width={Math.max(0, tx - pad.l)} height={axisY - 14} fill="var(--global-wash)" />
      <rect x={tx} y={14} width={Math.max(0, W - pad.r - tx)} height={axisY - 14} fill="var(--reg-wash)" />
      <text x={pad.l + 6} y={28} fontSize={fs} style={{ fill: 'var(--global)' }}>memory-bound</text>
      <text x={W - pad.r - 6} y={28} fontSize={fs} textAnchor="end" style={{ fill: 'var(--reg)' }}>compute-bound</text>
      <line x1={pad.l} x2={W - pad.r} y1={axisY} y2={axisY} stroke="var(--ink)" strokeWidth="1.4" />
      {[0.1, 1, 10, 100, 1000].map((v) => (
        <g key={v}>
          <line x1={X(v)} x2={X(v)} y1={axisY} y2={axisY + 5} stroke="var(--ink)" />
          <text x={X(v)} y={axisY + 18} fontSize={fs} textAnchor="middle">{v}</text>
        </g>
      ))}
      <text x={W / 2} y={H - 6} fontSize={fs + 0.5} textAnchor="middle">
        {narrow ? 'FLOP per byte of global memory, log scale' : 'operations per byte of global-memory traffic (FLOP/B), log scale'}
      </text>
      {ms.map((m, i) => {
        const cx = X(m.ratio);
        const ly = top + level[i] * step;
        const tone = regime(m.ratio, thr) === 'memory-bound' ? 'var(--global)' : 'var(--reg)';
        const anchor = cx < edge ? 'start' : cx > W - edge ? 'end' : 'middle';
        return (
          <g key={m.id} data-mark={m.id}>
            <line x1={cx} x2={cx} y1={ly + 5} y2={axisY} stroke={tone} strokeWidth={m.tiled ? 1.8 : 1} />
            <circle cx={cx} cy={axisY} r={m.tiled ? 6 : 4.5} fill={m.tiled ? tone : 'var(--page)'} stroke={tone} strokeWidth="1.8" />
            <text x={anchor === 'start' ? cx - 4 : anchor === 'end' ? cx + 4 : cx} y={ly} fontSize={fs} textAnchor={anchor} fontWeight={m.tiled ? 650 : 400} style={{ paintOrder: 'stroke', stroke: 'var(--page)', strokeWidth: 4, strokeLinejoin: 'round' }}>
              {m.label}
            </text>
          </g>
        );
      })}
      <line x1={tx} x2={tx} y1={8} y2={axisY + 30} stroke="var(--ink)" strokeWidth="2" strokeDasharray="5 3" />
      <text x={tx + (tx > W - 150 ? -6 : 6)} y={axisY + 42} fontSize={fs + 0.5} fontWeight="650" textAnchor={tx > W - 150 ? 'end' : 'start'} data-threshold-label>
        threshold {thrFmt(thr)}
      </text>
    </svg>
  );
}

function ThresholdBench() {
  const { progress } = useProgress();
  const narrow = useNarrow();
  const committed = typeof progress.answers[P_KNEE]?.value === 'number';
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
  const offered = DEVICES.filter((d) => d.key !== 'book2025dense');
  const dev = key === 'book2025tc' && dense ? DEVICES.find((d) => d.key === 'book2025dense')! : (offered.find((d) => d.key === key) ?? DEVICES[0]);
  const thr = threshold(dev.peak, dev.bw);
  const base = DEVICES[0];
  const baseThr = threshold(base.peak, base.bw);
  const need = tileFor(thr);
  const ms = marks(tile);
  return (
    <Bench
      title="Where the threshold went"
      task="Start on the 2007 column and find the first T that crosses. Then change the column."
      foot={
        <>
          Log scale. Marks are kernel ratios from earlier chapters; the filled one moves with T. Left of the dashed
          line a kernel is memory-bound on the chosen figures. {dev.note} Other devices, your own among them, are the
          roofline of{' '}
          <a href={href('ch05', '5.1')}>chapter 5, §5.1</a>.
        </>
      }
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
      <Strip thr={thr} tile={tile} narrow={narrow} />
      <div className="cols" style={{ marginTop: '0.8rem' }}>
        <div className="panel" data-readout="knee" aria-live="polite">
          <p style={{ fontSize: '1.4rem', margin: '0 0 0.2rem', fontWeight: 600 }}>threshold {thrFmt(thr)} FLOP/B</p>
          <p style={{ margin: '0 0 0.5rem' }} className="muted">
            {fmtInt(dev.peak)} GFLOP/s over {show(dev.bw)} GB/s
            {dev.key !== 'book2007' ? <>, {fmt(thr / baseThr, 2)} times the 2007 column's {thrFmt(baseThr)}</> : null}.
          </p>
          <p style={{ margin: '0 0 0.4rem' }}>
            Smallest T with 0.25 × T at or above it: <b data-need>{fmtInt(need)}</b>
            {need > 128 ? ', far off this slider' : ''}.
          </p>
          <p style={{ margin: 0 }}>
            At T = {tile} the tiled kernel reaches {fmt(0.25 * tile)} FLOP/B:{' '}
            {regime(0.25 * tile, thr) === 'memory-bound' ? <span className="chip global">memory-bound</span> : <span className="chip reg">compute-bound</span>}
          </p>
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
const THR_DENSE = threshold(DEVICES.find((d) => d.key === 'book2025dense')!.peak, BOOK_TC.bw);
const TENSOR_AUDIT = auditPrinted().find((a) => a.row === 'tensor')!;
const TENSOR_ROW = BOOK.rows.find((r) => r.id === 'tensor')!;
/** The 2007 figure that would make the printed 638x right against the printed 2200. */
const FROM_FOR_638 = TENSOR_ROW.to / TENSOR_AUDIT.printed;
/** Shared memory, in MB, for one T x T tile of M and one of N as 4-byte floats. */
export const tileBytesMB = (t: number) => (2 * t * t * 4) / 1e6;
const T_TC = tileFor(THR_TC);
const T_DENSE = tileFor(THR_DENSE);
const T_2007 = tileFor(THR_2007);

export const questions240: Record<string, Question> = {
  'ch24/start/q-compose': {
    kind: 'choice',
    id: 'ch24/start/q-compose',
    prompt: (
      <>
        Min-p sampling keeps every token whose probability is at least a fixed fraction of the most likely token's,
        then samples among them. The map lights five pattern chapters for sampling. Which does the min-p cut-off
        itself need?
      </>
    ),
    options: [
      {
        label: 'A sort (chapter 14), then a scan (chapter 11) of the sorted probabilities, as top-p does.',
        why: "Top-p needs them because its cut-off is a running sum over the probabilities in sorted order. Min-p's cut-off depends on one number, the largest probability, so no order is needed.",
      },
      {
        label: 'A reduction (chapter 10) for the largest probability, then a filter (chapter 12) to gather the tokens that pass.',
        correct: true,
      },
      {
        label: 'Nothing beyond step one: each thread compares its own probability with the threshold.',
        why: 'The threshold is a fraction of the largest probability in the row, and no thread knows that alone. Finding it is a reduction.',
      },
      {
        label: 'A histogram (chapter 9) of the probabilities.',
        why: 'A histogram counts items per bin. The cut-off needs one value, the maximum, not a count per bin.',
      },
    ],
    explain: (
      <p>
        Once a max reduction (chapter 10) has the largest probability, each token's test is elementwise, and gathering
        the survivors without holes is a filter (chapter 12), whose stable form runs a scan over keep flags. No sort is
        needed. Because every probability shares one normaliser, the test can even run on logits: keep a token if its
        logit is at least the largest logit plus the log of the fraction. Min-p is not in the book; module X1 builds
        samplers from these chapters.
      </p>
    ),
    source: 'ch. 24 §24.1, pp. 577–578, for the steps; ch. 10 (pp. 221–250) and ch. 12 (pp. 289–302) for the patterns; min-p is ours, not the book\'s',
  },
  ...Object.fromEntries(LIMITS.map((c) => [c.qid, limitQuestion(c.qid)])),
  'ch24/start/q-638': {
    kind: 'numeric',
    id: 'ch24/start/q-638',
    prompt: (
      <>
        Fig. 24.1 prints 0.345 TFLOPS for 2007, 2200 TFLOPS for the 2025 tensor-core figure, and 638x between them.
        Keeping the 2200, what 2007 throughput would make 638x correct?
      </>
    ),
    answer: Number(FROM_FOR_638.toFixed(2)),
    tol: 0.01,
    unit: 'TFLOPS',
    explain: (
      <p>
        2200 / 638 = {FROM_FOR_638.toFixed(2)} TFLOPS, ten times the printed 0.345. NVIDIA's own factors put the G80 at
        128 multiply-adds per clock × 2 × 1.35 GHz = 345.6 GFLOPS, and NVIDIA prints 2.2 petaFLOPS for the 2025 row. With
        both inputs confirmed, the ratio is the only cell left: it should read about {fmtX(TENSOR_AUDIT.recomputed)}.
      </p>
    ),
    source: 'ch. 24 §24.2, Fig. 24.1, p. 578; notes/today/ch24-facts.md, A2 and A9',
  },
  'ch24/start/q-tile': {
    kind: 'numeric',
    id: 'ch24/start/q-tile',
    prompt: (
      <>
        Against the book's 2025 tensor-core figure the threshold is {fmt(THR_TC)} FLOP/B, which chapter 5's tiled matmul
        reaches only at T = {fmtInt(T_TC)}. That kernel keeps one T × T tile of M and one of N in shared memory, as
        4-byte floats. How many megabytes would one block need?
      </>
    ),
    answer: Number(tileBytesMB(T_TC).toFixed(2)),
    // 9.68 MB, or 9.23 if counted in units of 2^20 bytes: both are right.
    tol: 0.05,
    unit: 'MB',
    explain: (
      <>
        <p>
          2 × {fmtInt(T_TC)} × {fmtInt(T_TC)} × 4 bytes = {fmt(tileBytesMB(T_TC))} MB per block. Compare that with the
          limit <code>python3 labs/pmpp.py doctor</code> prints for your device (<code>sharedMemPerBlock</code> on CUDA
          and HIP). On NVIDIA's dense TF32 figure T = {T_DENSE} still needs {fmt(tileBytesMB(T_DENSE), 2)} MB; on the
          2007 figures T = {T_2007} needed {fmt((tileBytesMB(T_2007) * 1e6) / 1024, 2)} KB.
        </p>
        <p>
          The reuse has to come from somewhere else: larger output tiles held in registers (chapter 15), batching, or
          fewer bytes per value.
        </p>
      </>
    ),
    source: 'our analysis with the §5.1 method (ch. 5, pp. 94–97) and the tiled kernel of §5.4 on the figures of Fig. 24.1, p. 578',
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
    metal: "None named in Metal documentation; Apple's spec page lists “Neural Accelerators” in the GPU, with no link to MSL stated",
    facts: 'A9, B1, C1',
  },
  {
    role: 'How kernel code reaches it',
    cuda: (
      <>
        CUDA Tile (<code>cuda::tiles</code> in <code>cuda_tile.h</code>, Toolkit 13.3 onward), or libraries. The book's
        §15.9 (p. 369) also names the WMMA and WGMMA instructions.
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
  const { progress } = useProgress();
  const committed = typeof progress.answers[P_HOSTLINK]?.value === 'number';
  // What the audit finds is discussed below it, so that text waits for its Check.
  const [audited, setAudited] = useState(false);
  return (
    <>
      <p>
        Chapter 24 is four pages with no code and no exercises: the book's goals revisited, then the authors' outlook.
        This page keeps three things: the three steps as a statement of what you can now do, the chapter's one table
        checked line by line, and a reading of that table for kernel work.
      </p>
      <Source pages="577" />

      <h2>Before you start</h2>
      <p>
        Five cards drawn from chapters 2, 5, 20 and 23. The book's last table has five rows, each a limit you already
        reason with: place each card on the row that limits it. Each card comes back in{' '}
        <a href={href('review')}>review</a>.
      </p>
      <LimitsBench />

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

      <h2>The book's one table</h2>
      <p>
        Section 24.2 opens with a table that sets the first CUDA GPU beside a 2025 one, with a ratio column. Before you
        see it, commit to a number for one of its rows.
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
      <TableAudit onCheck={() => setAudited(true)} />
      {audited ? (
        <>
          <p data-after-audit>
            The table contradicts itself twice. 2200 / 0.345 is 6,377, not 638, and the 2,200 figure is NVIDIA's own,
            so the ratio is the wrong cell. “1,800 TB/s” is 1,800 GB/s, and it counts both directions together while
            the PCIe row counts each way.
          </p>
          <Aside>
            <p>
              Facts A9 and A13 for these two, A9 and A6 for the next two, A3, A4, A8 and A11 for the rest
              (<code>notes/today/ch24-facts.md</code>).
            </p>
          </Aside>
          <p>
            Two more cells are wrong against NVIDIA's documents. The row labelled “Tensor Core FP32” is TF32 on Tensor
            Cores counted with sparsity; dense is half. The 1.5 GB board is PCI Express generation 1 in NVIDIA's board
            specification, so on those figures the host link grew about 16x, not 8x. The rest is looser: the 2007
            column takes its bandwidth from one board and its capacity from another, and the 2025 column's 80 TFLOPS
            and 192 GB match particular NVIDIA figures rather than the boards that ship, which have 75 TFLOPS and 180
            GB.
          </p>
          <Today checked="2026-11-20 (ch24-facts A2, A8, A11)">
            <p>
              Every figure is a published peak, not a measurement: 0.345 TFLOPS is 128 multiply-adds per clock × 2 ×
              1.35 GHz.
            </p>
          </Today>
          <p>
            One point survives every correction: everything on the device grew by about two orders of magnitude or
            more, the host link by one. Our reading (the book only prints the numbers): keep weights, the KV cache and
            sampling on the device.
          </p>
          <Ask q={questions240['ch24/start/q-638']} />
        </>
      ) : committed ? (
        <p className="muted" data-gate="audit">
          What the check finds, and a question on it, follow once you press Check above.
        </p>
      ) : null}

      <h2>Reading the table as a kernel writer</h2>
      <p>
        The book stops at the ratios. Section 5.1 gives the top two rows a better use: a device's threshold is its peak
        throughput over its peak bandwidth, and a kernel whose FLOP/B is below it is memory-bound. Applying that to Fig.
        24.1 is ours; chapter 24 does not do it. The strip puts kernels from earlier chapters on one FLOP/B axis. Choose
        a column and the threshold moves; everything left of it is memory-bound on those figures.
      </p>
      <Simplification>
        <p>
          Peaks are vendor figures, caches are ignored, and the two linear-layer marks are 16-bit kernels set against
          FP32 and TF32 peaks, so read their side of the line as indicative. Batch 1 is memory-bound on any peak.
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
        For matrix-matrix work, the next source of reuse after a shared-memory tile is chapter 15's register tiles,
        though against the tensor-core figure even the 128 × 128 tile falls short in this model. Batch-1 generation,
        about 1 FLOP/B in the book's 16-bit example (§20.6, p. 505), is memory-bound in every column; its ratio moves
        only with batching (chapters 20 and 22) or fewer bytes per weight (module X2). The arithmetic behind the larger
        compute number, TF32 on Tensor Cores (dense, so at most half the printed figure), is what{' '}
        <code>torch.backends.cuda.matmul.allow_tf32</code> switches on for matmul.
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
        backends this course builds for. The last column gives the items of <code>notes/today/ch24-facts.md</code>{' '}
        behind each row; WMMA and WGMMA are the book's own names (ch. 15 §15.9, p. 369).
      </p>
      <Beyond>
        <p>Chapter 24 names no other vendor. This table is ours.</p>
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
        be built. Where a cell says “not found” or “none documented”, we found none in the pages read; the vendor does
        not say there is none.
      </p>
      <Today checked="2026-11-20 (ch24-facts B1, B4, C1, C3, C5)">
        <p>AMD's MI355X page footnotes “TF32 support through software emulation”.</p>
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
          NCCL's public repository dates from 2015, so “new” is relative. The claims on job sizes and lithography are
          the authors' judgement.
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
