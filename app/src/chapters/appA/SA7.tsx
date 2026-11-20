import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  ARITH_LABEL,
  EPS_EXPONENTS,
  bookPanels,
  busyFraction,
  captionText,
  diffPanels,
  equation,
  printedA11,
  system,
  trace,
  type ArithId,
  type Phase,
  type SystemId,
} from '../../sim/gauss';
import { GaussScene, STATE_LABEL, Timeline, timelineWidth } from './GaussScene';

const minus = (s: string) => s.replace(/-/g, '−');
const cellText = (t: string) => (t === '0' ? '·' : minus(t));
const bad = (v: number) => !Number.isFinite(v);

// ---------------------------------------------------------------- the stepper
interface Config {
  sys: SystemId;
  pivot: boolean;
  arith: ArithId;
  eps: number;
}

function opFor(p: Phase, t: number): string {
  const s = p.threads[t];
  if (p.kind === 'divide' && p.divisors?.[t]) return '÷ ' + minus(p.divisors[t]!.text);
  if (p.kind === 'subtract' && s === 'working') return `− row ${p.col + 1}`;
  if (p.kind === 'pivot' && s === 'searching') {
    const j = p.pivot!.rows.indexOf(t);
    return `|${minus(p.pivot!.abs[j].text)}|` + (p.pivot!.winner === t ? ', the largest' : '');
  }
  if (p.kind === 'swap' && s === 'working') return `swaps with row ${(p.swap![0] === t ? p.swap![1] : p.swap![0]) + 1}`;
  if (p.kind === 'backsub' && s === 'working') return 'substitutes';
  return '';
}

function GaussStepper({ title, task, init }: { title: string; task: string; init: Config }) {
  const [cfg, setCfg] = useState<Config>(init);
  const [view, setView] = useState<'matrix' | 'equations'>('matrix');
  const [i, setI] = useState(0);
  const sys = useMemo(() => system(cfg.sys, cfg.eps), [cfg.sys, cfg.eps]);
  const t = useMemo(() => trace(sys, { pivot: cfg.pivot, arith: cfg.arith }), [sys, cfg.pivot, cfg.arith]);
  useEffect(() => setI(0), [t]);
  const last = t.phases.length - 1;
  const at = Math.min(i, last);
  const ph = t.phases[at];
  const n = t.n;
  const anyBad = ph.m.some((r) => r.some((c) => bad(c.v)));
  const upTo = { ...t, phases: t.phases.slice(0, at + 1) };
  const busySoFar = upTo.phases.some((p) => p.kind === 'divide' || p.kind === 'subtract') ? busyFraction(upTo) : null;
  const set = (patch: Partial<Config>) => setCfg((c) => ({ ...c, ...patch }));
  const lane = 30;
  const tw = timelineWidth(t.phases);
  const exactSol = sys.solution;
  const done = ph.kind === 'done';
  const exact = useMemo(() => trace(sys, { pivot: true, arith: 'exact' }).solution, [sys]);
  // Largest relative error of the computed solution; above 10% the answer counts as wrong.
  const relErr = Math.max(...t.solution.map((c, k) => Math.abs(c.v - exact[k].v) / Math.max(1e-30, Math.abs(exact[k].v))));
  const close = t.finite && relErr <= 0.1;
  return (
    <Bench
      title={title}
      task={task}
      foot={
        <>
          Rows are numbered as in the book (row 1 at the top); thread t owns row t + 1. Exact fractions keep infinity and NaN
          as IEEE arithmetic would; FP32 and FP16 round every operation to nearest even. In the timeline, dark cells are
          threads working, dashed cells threads skipping, grey cells idle threads, and the vertical bars are barriers.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          System
          <select value={cfg.sys} onChange={(e) => set({ sys: e.target.value as SystemId })} aria-label="system">
            <option value="book1">The book's first system</option>
            <option value="book2">The book's second (no X in row 1)</option>
            <option value="tiny">2 × 2 with a tiny pivot ε (ours)</option>
          </select>
        </label>
        {cfg.sys === 'tiny' ? (
          <label className="field">
            ε = 1e−{cfg.eps}
            <input type="range" min={EPS_EXPONENTS[0]} max={EPS_EXPONENTS[EPS_EXPONENTS.length - 1]} value={cfg.eps} aria-label="epsilon exponent"
              onChange={(e) => set({ eps: +e.target.value })} />
          </label>
        ) : null}
        <label className="check">
          <input type="checkbox" checked={cfg.pivot} onChange={(e) => set({ pivot: e.target.checked })} /> pivoting
        </label>
        <label className="field">
          Arithmetic
          <span className="seg" role="group" aria-label="arithmetic">
            {(['exact', 'fp32', 'fp16'] as ArithId[]).map((a) => (
              <button key={a} aria-pressed={cfg.arith === a} onClick={() => set({ arith: a })}>{ARITH_LABEL[a]}</button>
            ))}
          </span>
        </label>
        <label className="field">
          View
          <span className="seg" role="group" aria-label="view">
            {(['matrix', 'equations'] as const).map((v) => (
              <button key={v} aria-pressed={view === v} onClick={() => setView(v)}>{v}</button>
            ))}
          </span>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setI(0)} disabled={at === 0}>Restart</button>
        <button onClick={() => setI(at - 1)} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => setI(at + 1)} disabled={at >= last}>Step</button>
        <button onClick={() => setI(last)} disabled={at >= last}>Run to the end</button>
        <span className="muted" aria-live="polite" data-at={at} data-total={last}>phase {at} of {last}: {ph.kind === 'backsub' ? 'back substitution' : ph.kind}</span>
      </div>
      <div className={'narrate' + (anyBad ? ' bad' : '')} role="status" style={{ marginBottom: '0.8rem' }} data-phase={ph.kind}>
        {ph.note}
        {ph.kind === 'pivot' ? (
          <>
            <br />
            <span className="muted">
              Max reduction:{' '}
              {ph.pivot!.levels
                .map((lvl) => lvl.map((r) => `|${minus(ph.pivot!.abs[ph.pivot!.rows.indexOf(r)].text)}|`).join(' '))
                .join('  →  ')}
            </span>
          </>
        ) : null}
      </div>
      {ph.kind === 'barrier' ? (
        <div data-barrier style={{ background: 'var(--ink)', color: 'var(--page)', borderRadius: 3, padding: '0.25rem 0.7rem', marginBottom: '0.4rem', fontFamily: 'var(--sans)', fontSize: '0.82rem' }}>
          barrier: all {n} threads wait here
        </div>
      ) : null}
      <div style={{ overflowX: 'auto' }}>
      <table className="threads" aria-label="Threads and their rows" data-view={view}>
        <thead>
          <tr>
            <th>thread</th>
            <th>state</th>
            <th>this phase</th>
            <th>{view === 'matrix' ? `row: ${sys.vars.join(', ')}, right-hand side` : 'equation'}</th>
          </tr>
        </thead>
        <tbody>
          {ph.m.map((row, k) => (
            <tr key={k} data-thread={k} className={ph.threads[k] === 'working' ? 'cur' : ''}>
              <td>{k} · row {k + 1}</td>
              <td data-state={ph.threads[k]}>{STATE_LABEL[ph.threads[k]]}</td>
              <td>{opFor(ph, k)}</td>
              <td>
                {view === 'matrix' ? (
                  <span className="mx" style={{ gridTemplateColumns: `repeat(${n + 1}, auto)` }}>
                    {row.map((c, j) => (
                      <span key={j} className={bad(c.v) ? 'bad' : ''} data-cell={`${k},${j}`} style={j === n ? { borderLeftWidth: 3 } : undefined}>
                        {cellText(c.text)}
                      </span>
                    ))}
                  </span>
                ) : (
                  <span className={row.some((c) => bad(c.v)) ? 't-fail' : ''} data-eq={k}>{equation(row.map((c) => c.text), sys.vars)}</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      </div>
      <div style={{ marginTop: '0.8rem', overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${tw + 76} ${n * lane + 8}`} style={{ width: (tw + 76) * 1.3, maxWidth: '100%', display: 'block' }} role="img" aria-label="Timeline of thread states">
          {Array.from({ length: n }, (_, k) => (
            <text key={k} x={0} y={8 + k * lane + 14} fontSize="11">thread {k}</text>
          ))}
          <Timeline phases={t.phases.slice(0, at + 1)} n={n} x={66} y={8} lane={lane} />
        </svg>
      </div>
      <p className="muted" style={{ margin: '0.5rem 0 0' }} data-busy={busySoFar ?? ''}>
        {busySoFar == null
          ? 'No divide or subtract step has run yet.'
          : `Thread slots doing work in the divide and subtract steps so far: ${Math.round(busySoFar * 100)}%.`}
      </p>
      {done ? (
        <p style={{ margin: '0.6rem 0 0' }} data-solution={t.solution.map((c) => c.text).join(',')}>
          {close ? (
            <span className="chip ok">{relErr === 0 ? 'exact solution' : `close: largest error ${Number((relErr * 100).toPrecision(2))}%`}</span>
          ) : (
            <span className="chip fail">{t.finite ? 'wrong answer' : 'no answer: NaN or infinity'}</span>
          )}{' '}
          Computed: {sys.vars.map((v, k) => `${v} = ${cellText(t.solution[k].text)}`).join(', ')}. Exact: {sys.vars.map((v, k) => `${v} = ${minus(exactSol[k])}`).join(', ')}.
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the erratum hunt
function ErratumHunt() {
  const computed = useMemo(() => bookPanels('A.11'), []);
  const diffs = useMemo(() => diffPanels(printedA11, computed), [computed]);
  const wrong = useMemo(
    () => new Set(diffs.map((d) => `${printedA11.findIndex((p) => p.label === d.panel)}:${d.what === 'caption' ? 'cap' : `${d.row}:${d.col}`}`)),
    [diffs],
  );
  const [marks, setMarks] = useState<Set<string>>(new Set());
  const [revealed, setRevealed] = useState(false);
  const [showComputed, setShowComputed] = useState(false);
  const toggle = (k: string) => {
    if (revealed) return;
    setMarks((m) => {
      const x = new Set(m);
      if (x.has(k)) x.delete(k);
      else x.add(k);
      return x;
    });
  };
  const found = [...marks].filter((k) => wrong.has(k)).length;
  const falseAlarms = [...marks].filter((k) => !wrong.has(k)).length;
  const panels = showComputed ? computed : printedA11;
  return (
    <Bench
      title="Fig. A.11 as printed"
      task="Click every caption or number you think the printed figure gets wrong, then check."
      foot="Transcribed from p. 598 (book notes); captions shortened to their content. Rows are numbered from 1 and blank positions drawn as dots, as in the book."
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button className="primary" onClick={() => setRevealed(true)} disabled={revealed}>Check my marks</button>
        <button onClick={() => { setMarks(new Set()); setRevealed(false); setShowComputed(false); }}>Clear</button>
        {revealed ? (
          <label className="check">
            <input type="checkbox" checked={showComputed} onChange={(e) => setShowComputed(e.target.checked)} /> show the computed figure
          </label>
        ) : null}
        <span className="muted" data-marks={marks.size}>{marks.size} marked</span>
      </div>
      {revealed ? (
        <p data-result={`${found},${wrong.size},${falseAlarms}`} style={{ margin: '0 0 0.8rem' }}>
          <span className={found === wrong.size ? 'chip ok' : 'chip fail'}>found {found} of {wrong.size}</span>{' '}
          {falseAlarms ? <span className="chip fail">{falseAlarms} marked but right</span> : null} The printed Step 1 divides
          row 1 by 3, but row 1 is [2 3 1 11] and the result shown is a division by 2. Steps 5 and 6 print row 1 of the first
          system's Fig. A.10 (1, 5/3, 2/3, 19/3) instead of this system's (1, 3/2, 1/2, 11/2). The answer, (1, 2, 3), is right.
        </p>
      ) : null}
      <div style={{ display: 'grid', gap: '0.8rem', gridTemplateColumns: 'repeat(auto-fill, minmax(13.5rem, 1fr))' }}>
        {panels.map((p, i) => {
          const capKey = `${i}:cap`;
          const capCls = revealed ? (wrong.has(capKey) ? (marks.has(capKey) ? 'found' : 'missed') : marks.has(capKey) ? 'false' : '') : '';
          return (
            <div key={p.label} className="panel" data-panel={p.label}>
              <div className="mx-label">{p.label}{p.solved ? <span className="muted"> · solves {p.solved}</span> : null}</div>
              {p.caption.op !== 'none' ? (
                <button className="quiet" aria-pressed={marks.has(capKey)} data-key={capKey} data-verdict={capCls}
                  onClick={() => toggle(capKey)}
                  style={{ fontSize: '0.8rem', marginBottom: '0.45rem', textAlign: 'left', ...(capCls === 'found' || capCls === 'missed' ? { borderColor: 'var(--fail)', color: capCls === 'missed' ? 'var(--fail)' : undefined } : {}) }}>
                  {captionText(p.caption)}
                </button>
              ) : (
                <div className="muted" style={{ fontSize: '0.8rem', marginBottom: '0.45rem' }}>the system</div>
              )}
              <span className="mx" style={{ gridTemplateColumns: 'repeat(4, auto)', display: 'inline-grid' }}>
                {p.rows.flatMap((row, r) =>
                  row.map((v, c) => {
                    const k = `${i}:${r}:${c}`;
                    const isWrong = wrong.has(k);
                    const marked = marks.has(k);
                    const cls = revealed ? (isWrong ? (showComputed ? 'good' : 'bad') : marked ? 'void' : '') : '';
                    return (
                      <span key={k} role="button" tabIndex={0} data-key={k} data-marked={marked ? 1 : 0} className={cls}
                        onClick={() => toggle(k)} onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && toggle(k)}
                        style={{ cursor: revealed ? 'default' : 'pointer', ...(marked ? { outline: '2px solid var(--ink)', outlineOffset: -2 } : {}) }}>
                        {cellText(v)}
                      </span>
                    );
                  }),
                )}
              </span>
            </div>
          );
        })}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSA7: Record<string, Question> = {
  'appA/A.7/q-idle': {
    kind: 'numeric',
    id: 'appA/A.7/q-idle',
    prompt: (
      <>
        8 equations, one thread per row, the book's design. Over the divide and subtract steps of the elimination, what
        percentage of thread slots does useful work?
      </>
    ),
    answer: 53.33,
    tol: 0.01,
    unit: '%',
    explain: (
      <p>
        Divide step k keeps n − k threads working and subtract step k keeps n − k − 1, so the n + (n − 1) = 15 steps of 8
        slots hold 8 + 7 + 7 + 6 + ... + 1 = 64 = n² busy slots out of 120: 53.3%. In general the share is n / (2n − 1),
        which falls toward a half as n grows: 1,024 equations give 1,024 / 2,047. Every shrinking-work algorithm in this
        book has this shape, and it is why chapter 10's reduction tree gets coarsened.
      </p>
    ),
    source: 'book §A.7, p. 597 (the design); our count',
  },
  'appA/A.7/q-tiny': {
    kind: 'choice',
    id: 'appA/A.7/q-tiny',
    prompt: (
      <>
        FP32, no pivoting, the system [[ε, 1], [1, 1]] x = [1, 2] with ε = 10<sup>−8</sup>. The true x1 is about 1. What
        does elimination give for x1?
      </>
    ),
    options: [
      { label: '0, because 1 − 10⁸ and 2 − 10⁸ round to the same number.', correct: true },
      { label: 'About 1, correct to about seven digits, as FP32 promises.', why: 'Each operation is accurate to half an ULP; the algorithm amplifies the errors.' },
      { label: 'NaN, because ε rounds to 0 in FP32.', why: '10⁻⁸ is far above FP32\'s smallest normal number, about 1.2 × 10⁻³⁸. In FP16 it would round to 0.' },
      { label: '10⁸, the reciprocal of the tiny pivot.', why: 'That is the coefficient after the division, not the solution.' },
    ],
    explain: (
      <p>
        Row 1 divided by ε holds 10<sup>8</sup> twice. Subtracting it from row 2 gives 1 − 10<sup>8</sup> and 2 −
        10<sup>8</sup>, and near 10<sup>8</sup> FP32 values are 8 apart, so both round to −10<sup>8</sup>: x2 = 1 exactly.
        Back substitution then computes x1 = 10<sup>8</sup> − 10<sup>8</sup> · 1 = 0. Swap the rows first and the same
        steps give x1 = 1.
      </p>
    ),
    source: 'our example of the book\'s "subtler case", §A.7, p. 597; checked by src/sim/gauss.test.ts',
  },
  'appA/A.7/q-a11': {
    kind: 'choice',
    id: 'appA/A.7/q-a11',
    prompt: (
      <>
        Fig. A.11's Step 1 caption says to divide row 1 by 3. After the swap, row 1 is [2 3 1 11], and the figure shows
        the result [1 3/2 1/2 11/2]. What should the divisor be?
      </>
    ),
    options: [
      { label: '3, row 1\'s coefficient of Y.', why: 'Step 1 eliminates X, so the divisor is the coefficient of X.' },
      { label: '2, row 1\'s coefficient of X.', correct: true },
      { label: '5, the largest coefficient left in the matrix.', why: 'Pivoting chose the row; the division uses that row\'s own coefficient.' },
      { label: '1, since the swap already normalised the row.', why: 'A swap moves a row; it does not scale it.' },
    ],
    explain: (
      <p>
        Dividing [2 3 1 11] by 2 gives [1 3/2 1/2 11/2], exactly the printed result; dividing by 3 would give [2/3 1 1/3
        11/3]. The caption is a slip; our guess is that it was carried over from Fig. A.10, whose first row does divide by
        3.
      </p>
    ),
    source: 'book §A.7, p. 598, Fig. A.11; notes/appA.md "Unclear in the PDF"',
  },
  'appA/A.7/q-pivot-cost': {
    kind: 'choice',
    id: 'appA/A.7/q-pivot-cost',
    prompt: <>Why is pivoting cheap inside one block and expensive across many blocks?</>,
    options: [
      {
        label: 'Each column needs a max over all rows that every thread sees: a barrier in a block, a grid sync across blocks.',
        correct: true,
      },
      { label: 'Swapping two rows in global memory takes far longer than swapping them in shared memory.', why: 'Moving one row is cheap; the cost is that everyone must agree on which row, once per column.' },
      { label: 'Thread blocks cannot run a max reduction on their own, so the search moves to the host.', why: 'They can (chapter 10); the problem is combining the blocks\' answers before anyone may divide.' },
      { label: 'Pivoting roughly doubles the arithmetic, and many blocks make that cost visible.', why: 'A search adds about n comparisons per column, small next to the elimination itself.' },
    ],
    explain: (
      <p>
        Inside a block the search is a shared-memory reduction followed by a barrier. Blocks cannot wait for each other in
        an ordinary launch, so across blocks every column needs a grid-wide synchronisation (a cooperative launch) or a
        kernel boundary: a launch per column. That is the communication the book's reference [6] is about avoiding.
      </p>
    ),
    source: 'book §A.7, pp. 598–599; chapter 10 §10.9 and notes/today/ch10-facts.md A8',
  },
  'appA/A.7/q-partial': {
    kind: 'choice',
    id: 'appA/A.7/q-partial',
    prompt: <>cuSOLVER's getrf documents "LU with partial pivoting". What does it do before eliminating each column?</>,
    options: [
      { label: 'Swaps rows so that the pivot is the largest-magnitude entry of that column among the remaining rows.', correct: true },
      { label: 'Swaps rows and columns to bring the largest entry of the remaining submatrix to the pivot.', why: 'That is complete pivoting, A = P L U Q (LAPACK getc2).' },
      { label: 'Searches only a local group of nearby rows, so less data has to be exchanged.', why: 'That is the book\'s description of partial pivoting, which matches neither library term.' },
      { label: 'Nothing, unless the pivot is exactly zero; then it swaps with the next row.', why: 'It searches every column; a tiny non-zero pivot is the dangerous case too.' },
    ],
    explain: (
      <p>
        In LAPACK, cuSOLVER, rocSOLVER and Metal Performance Shaders, partial pivoting means row interchanges that choose
        the pivot within the current column, giving A = P L U. Swapping columns as well is complete pivoting. cuSOLVER
        also says plainly that its no-pivoting mode "is not numerically stable".
      </p>
    ),
    source: 'notes/today/appA-facts.md A15 (LAPACK dgetrf and dgetc2, cuSOLVER, rocSOLVER, MPS)',
  },
};

// ---------------------------------------------------------------- the section
export function SA7() {
  return (
    <>
      <p>
        This page is condensed. The book's §A.7 solves small linear systems by Gaussian elimination, sketches a parallel
        version, and shows how a zero or tiny divisor breaks it. Three things transfer to inference kernels: a
        row-per-thread design held together by a barrier after every step, the cost of threads going idle as the work
        shrinks, and the habit of finding the largest value before dividing by anything. There is no lab: for a real
        solver you call a library, and the end of the page says which.
      </p>
      <Source section="A.7" pages="594–599" />

      <h2>Stable and unstable</h2>
      <p>
        For a sum, the order of the additions changes the low bits (§A.6). For a linear solver it can decide whether you
        get an answer at all: some inputs need the operations in a different order before a solution appears. The book
        calls an algorithm numerically stable if it can find a suitable order, and so a solution, for every input that has
        one, and unstable otherwise. Stability requirements are one reason efficient parallel solvers are hard to find.
      </p>
      <Source section="A.7" pages="594" />

      <h2>Elimination, one row per thread</h2>
      <p>
        The book's example is three equations, 3X + 5Y + 2Z = 19, 2X + 3Y + Z = 11 and X + 2Y + 2Z = 11: three planes that
        meet at one point. Divide each equation by its coefficient of X, subtract the first from the other two, and X is
        gone from both. Repeat for Y on the last two, and the third reads 3Z = 9. Back substitution then runs upward: Z
        into the second equation gives Y, Y and Z into the first give X. Written as the rows of an augmented matrix, with
        the right-hand side as a last column, each operation on an equation is an operation on a row.
      </p>
      <Source section="A.7" pages="594–596" figs="Figs. A.9 and A.10" />
      <p>
        The parallel design gives each row to one thread, which does all the work on it, and keeps the matrix in one
        block's shared memory when it fits. The threads step through the columns together. After each division comes a
        barrier, because the subtraction reads the top row, which another thread has just written. After each subtraction
        comes another, so that the next division sees the result, and one more thread is left with nothing to do until
        back substitution. The book describes this in prose only; it gives no kernel.
      </p>
      <Source section="A.7" pages="597" />
      <GaussScene />
      <p>
        Now step through it yourself. Start with the book's first system in the matrix view, and compare each phase with
        Fig. A.10; switch to equations for Fig. A.9. Then try FP32 and FP16 on the same system and watch the answer drift a
        little from (1, 2, 3).
      </p>
      <Simplification>
        <p>
          One block, one thread per row, no timing and no warps; the rows fit in shared memory, as the book assumes. A
          pivot search is one phase here; on a GPU it is a reduction of log2 n rounds with barriers. A multiply and a
          subtract round separately (no fused multiply-add).
        </p>
      </Simplification>
      <GaussStepper
        title="Elimination, one phase at a time"
        task="Step one phase at a time. Where do the barriers fall, and which threads have nothing to do?"
        init={{ sys: 'book1', pivot: false, arith: 'exact', eps: 8 }}
      />
      <p>
        Count the cells in the timeline. On three rows, 9 of the 15 thread slots in the divide and subtract steps do work.
        Count how many rows each step keeps busy, then work out what that share becomes on a large system.
      </p>
      <Ask q={questionsSA7['appA/A.7/q-idle']} />

      <h2>When the divisor is zero, or tiny</h2>
      <p>
        Replace the first equation by 5Y + 2Z = 16. The system still has the solution (1, 2, 3), but step 1 asks thread 0
        to divide by its coefficient of X, which is now 0. In IEEE arithmetic nothing stops (§A.3): 5 / 0 is infinity, 0 /
        0 is NaN, and at the next subtraction every row picks up a NaN. The simple algorithm is unstable.
      </p>
      <Source section="A.7" pages="597" />
      <p>
        The subtler case is a lead coefficient that is not zero but small, and already carries rounding error from earlier
        steps. Dividing by it magnifies that error, and every later step inherits it, which is why instability is often
        measured as the growth of error from step to step.
      </p>
      <Source section="A.7" pages="597" />
      <p>
        A two-equation example makes it concrete: [[ε, 1], [1, 1]] x = [1, 2], whose solution x1 = 1 / (1 − ε) is about 1
        for small ε. In FP32 with ε = 10<sup>−8</sup> and no pivoting, the first row divided by ε holds 10<sup>8</sup>, and
        subtracting it from the second leaves 1 − 10<sup>8</sup> and 2 − 10<sup>8</sup>. Near 10<sup>8</sup>, FP32 values
        are 8 apart, so both round to the same number, and the second equation's information is gone.
      </p>
      <Beyond>
        <p>The 2 × 2 system is ours: a standard illustration of the book's "subtler case".</p>
      </Beyond>
      <GaussStepper
        title="Pivot or fail"
        task="Run it to the end, then change one thing at a time: the arithmetic, ε, pivoting."
        init={{ sys: 'tiny', pivot: false, arith: 'fp32', eps: 8 }}
      />
      <p>
        With exact fractions the same steps give x1 = 100,000,000 / 99,999,999: the algorithm is exact, and the rounding
        does the damage. In FP16 the trouble starts much earlier, at ε = 10<sup>−4</sup>; from 10<sup>−5</sup> the division
        by ε overflows to infinity, and from 10<sup>−8</sup> down ε itself rounds to 0, which turns the tiny pivot into
        the zero pivot above. Then select the book's second system
        without pivoting and step to the first subtraction.
      </p>
      <Ask q={questionsSA7['appA/A.7/q-tiny']} />

      <h2>Pivoting</h2>
      <p>
        The book's fix: before dividing, pick a remaining equation whose coefficient of the current variable is not zero,
        preferably the largest in magnitude, and swap it to the top. For the second system, 2X + 3Y + Z = 11 moves to the
        top, and the equation with no X becomes the second; its thread skips both the division and the subtraction for X,
        since there is nothing to eliminate. That skip is a branch some threads take and others do not: divergence inside a
        warp (chapter 4).
      </p>
      <Source section="A.7" pages="597–598" figs="Fig. A.11" />
      <p>
        Tick "pivoting" in either stepper and run the second system and the 2 × 2 system again. One combination still
        fails: FP16 with ε from 10<sup>−5</sup> to 10<sup>−7</sup>, pivoting on.
      </p>
      <Beyond>
        <p>
          Ours: the book's design divides every row by its own lead coefficient, including the row holding ε, and 1 / ε
          then exceeds FP16's largest value, 65,504. LU with partial pivoting divides each row's entry by the pivot instead,
          so every multiplier is at most 1 in magnitude.
        </p>
      </Beyond>

      <h3>Fig. A.11, as printed</h3>
      <p>
        The book draws the pivoting run of the second system as nine panels. Two things in the printed figure are wrong,
        one of them twice. Find them before reading on: mark every caption or number you believe is wrong, then check. The
        stepper above, set to the second system with pivoting and exact fractions, produces the right panels.
      </p>
      <ErratumHunt />
      <Ask q={questionsSA7['appA/A.7/q-a11']} />

      <h2>What pivoting costs in parallel</h2>
      <p>
        Pivoting starts every column with a question about all the remaining rows: which one holds the largest
        coefficient? With one thread per row that is a max reduction across threads, chapter 10's tree, followed by a swap
        that every thread must see. Inside one block that is affordable: a reduction in shared memory and a barrier per
        column, with some care to keep divergence within warps down.
      </p>
      <Source section="A.7" pages="598–599" />
      <p>
        Across blocks it is not. Blocks cannot wait for each other inside an ordinary launch (
        <a href={href('ch10', '10.9')}>§10.9</a>), so each column needs a grid-wide synchronisation, available only under a
        cooperative launch, or a kernel boundary: one launch per column. Across the nodes of a cluster it is worse again.
        The book points to communication-avoiding algorithms, which restructure the work to exchange less, and to
        randomisation, which researchers have found tends to keep accuracy high. It names both, and so does this page.
      </p>
      <Today checked="2026-11-20 (notes/today/ch10-facts.md A8 and its backend table)">
        <p>
          CUDA and HIP have <code>grid.sync()</code> under a cooperative launch whose grid fits on the device at once.
          Metal has no grid-wide barrier: use a second dispatch.
        </p>
      </Today>
      <Ask q={questionsSA7['appA/A.7/q-pivot-cost']} />

      <h2>What the libraries call it</h2>
      <p>
        The book defines partial pivoting as restricting the candidates to a localised set of equations, at a slight cost
        in accuracy, and its general rule swaps columns as well as rows. The libraries use the words differently. Partial
        pivoting there means row interchanges that choose the pivot within the current column, the factorisation A = P L
        U; swapping rows and columns is complete pivoting, A = P L U Q. The book's localised set matches neither. Our
        reading is that it describes a communication-avoiding variant, the subject of its reference [6]; we have not
        checked that.
      </p>
      <Today checked="2026-11-20 (notes/today/appA-facts.md A15; infra-facts.md C11)">
        <p>
          cuSOLVER <code>getrf</code>, rocSOLVER <code>getrf</code> (and <code>getrf_npvt</code> without pivoting), Metal's{' '}
          <code>MPSMatrixDecompositionLU</code> and Accelerate's LAPACK on the CPU all do LU with partial pivoting in the
          library sense. cuSOLVER: LU without pivoting "is not numerically stable".
        </p>
      </Today>
      <Ask q={questionsSA7['appA/A.7/q-partial']} />
      <p>
        For a linear system in your own code you call one of these; you do not write it. What you keep from this section
        is the pattern, and it turns up in inference kernels under other names.
      </p>

      <h2>The same move in a softmax</h2>
      <p>
        A safe softmax subtracts the row's maximum before calling exp, so no exponential overflows. Pivoting finds the
        largest divisor before dividing, so no division blows up. MX block scaling (<a href={href('appA', 'A.5')}>§A.5</a>)
        finds the largest magnitude before choosing the scale. All three run a max reduction first, so that the dangerous
        operation only ever sees magnitudes it can handle.
      </p>
      <Beyond>
        <p>This bridge is ours; the book does not mention softmax.</p>
      </Beyond>

      <h2>What this page leaves out</h2>
      <p>
        No solver lab, no LU kernel, and nothing beyond a name for communication-avoiding algorithms and randomised
        pivoting (p. 599). The book's reference [6] (Ballard, Demmel, Holtz and Schwartz) is the place to start for the
        first.
      </p>
    </>
  );
}
