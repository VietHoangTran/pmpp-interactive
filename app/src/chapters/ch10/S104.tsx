import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { STATUS, useLabStatus } from './status';
import {
  BOOK_PRINTED,
  CONVERGENT_SPEC,
  EXERCISE_6,
  FIG_VALUES,
  KIND_LABEL,
  MIRROR_SPEC,
  OWNED,
  PARTNERS,
  TESTS,
  WRITERS,
  iterations,
  runSpec,
  type InPlaceKind,
  type Iteration,
  type OwnedKey,
  type PartnerKey,
  type Spec,
  type SpecRun,
  type TestKey,
  type Trace,
  type WriterKey,
} from '../../sim/reduction';
import { functionSource } from '../../sim/toolchain';
import { widthChoices } from '../../sim/warps';

import { NoteGroup } from './NoteGroup';

// The lab whose runs give the status of the listings taken from its files (and from labs/ch10/shared/, which it runs).
const LAB10_1 = 'ch10/lab1_sum';
const TIERS: Record<string, Tier> = { input: 'global', output: 'global', row: 'global', i: 'reg', stride: 'reg', t: 'reg', b: 'reg', bdim: 'reg' };
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const missing = (s: string | null) => !s || s.startsWith('// could not load');
const pct = (x: number) => `${Math.round(x * 100)}%`;
const SIZES = [16, 256, 512, 1024, 2048];

/** Lines of `b` (1-based) that differ from the same line of `a`, ignoring the function's name. */
function changedLines(a: string, b: string, nameA: string, nameB: string): number[] {
  const la = a.split('\n');
  return b.split('\n').flatMap((l, i) => (l.replace(nameB, nameA) !== la[i] ? [i + 1] : []));
}

// ---------------------------------------------------------------- the 256-element table
function BookTable() {
  const s = iterations('simple', 256, 32);
  const c = iterations('convergent', 256, 32);
  const div = (d: number) => <td className={'num' + (d ? ' t-fail' : '')}>{d}</td>;
  return (
    <table className="data" data-part="book-table" style={{ maxWidth: '40rem', whiteSpace: 'nowrap' }}>
      <thead>
        <tr>
          <th>Iteration</th>
          <th className="num">Active threads</th>
          <th className="num">Simple: stride</th>
          <th className="num">warps</th>
          <th className="num">divergent</th>
          <th className="num">Convergent: stride</th>
          <th className="num">warps</th>
          <th className="num">divergent</th>
        </tr>
      </thead>
      <tbody>
        {s.iterations.map((it, k) => (
          <tr key={k}>
            <td>{it.n}</td>
            <td className="num">{it.active}</td>
            <td className="num">{it.stride}</td>
            <td className="num">{it.activeWarps}</td>
            {div(it.divergentWarps)}
            <td className="num">{c.iterations[k].stride}</td>
            <td className="num">{c.iterations[k].activeWarps}</td>
            {div(c.iterations[k].divergentWarps)}
          </tr>
        ))}
        <tr>
          <td><b>Total</b></td>
          <td className="num"><b>{s.results}</b></td>
          <td />
          <td className="num" data-total="simple" colSpan={2}><b>{s.slots / 32} × 32 = {s.slots}</b></td>
          <td />
          <td className="num" data-total="convergent" colSpan={2}><b>{c.slots / 32} × 32 = {c.slots}</b></td>
        </tr>
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------- the warp map
/** One iteration of one kernel: a row of lanes per warp, active lanes filled. */
function WarpRows({ it, width }: { it: Iteration; width: number }) {
  const cw = Math.max(4, Math.min(13, Math.floor(360 / width)));
  const rh = Math.max(8, cw + 3);
  const left = 34;
  const chipW = 62;
  const W = left + width * cw + chipW;
  const H = it.warps.length * rh + 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="One row per warp, one cell per lane; filled lanes are active" data-part="warps"
      style={{ width: W, maxWidth: '100%', display: 'block' }}>
      <defs>
        <pattern id="s104-pad" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="4" stroke="var(--ink-3)" strokeWidth="1" />
        </pattern>
      </defs>
      {it.warps.map((w) => (
        <g key={w.warp} data-warp={w.warp} data-state={w.state}>
          <text x={left - 5} y={w.warp * rh + rh - 3} fontSize={Math.min(10, rh)} textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
            w{w.warp}
          </text>
          {w.lanes.map((l, i) => (
            <rect key={i} x={left + i * cw} y={w.warp * rh + 1} width={cw - 1} height={rh - 2} rx="1"
              fill={l === null ? 'url(#s104-pad)' : l ? 'var(--ink-2)' : 'var(--cell)'} stroke="var(--rule)" strokeWidth="0.5" />
          ))}
          <text x={left + width * cw + 6} y={w.warp * rh + rh - 3} fontSize={Math.min(10, rh)}
            style={{ fill: w.state === 'divergent' ? 'var(--fail)' : w.state === 'full' ? 'var(--ok)' : 'var(--ink-3)', fontWeight: w.state === 'idle' ? 400 : 650 }}>
            {w.state}
          </text>
        </g>
      ))}
    </svg>
  );
}

/** Totals over the first n iterations. */
function soFar(t: Trace, n: number) {
  const its = t.iterations.slice(0, n);
  const slots = its.reduce((a, i) => a + i.slots, 0);
  const results = its.reduce((a, i) => a + i.active, 0);
  return { slots, results, efficiency: slots ? results / slots : 0 };
}

function KernelPanel({ t, n }: { t: Trace; n: number }) {
  const it = t.iterations[n - 1];
  const sf = soFar(t, n);
  return (
    <div className="panel" data-kernel={t.kind} data-iteration={n} style={{ minWidth: 0, overflowX: 'auto' }}>
      <div className="mx-label">
        {KIND_LABEL[t.kind]}: iteration {it.n}, stride {it.stride}
      </div>
      <p className="muted" style={{ margin: '0 0 0.5rem' }} data-readout="iteration">
        {it.active} active thread{it.active === 1 ? '' : 's'} in {it.activeWarps} of {t.warpCount} warp{t.warpCount === 1 ? '' : 's'};{' '}
        <span data-divergent={it.divergentWarps}>{it.divergentWarps} divergent</span>.
      </p>
      <WarpRows it={it} width={t.width} />
      <table className="threads" style={{ marginTop: '0.7rem' }} aria-label={`Iterations of the ${t.kind} kernel`}>
        <thead>
          <tr><th>it.</th><th>stride</th><th>active threads</th><th>active warps</th><th>divergent</th><th>slots</th></tr>
        </thead>
        <tbody>
          {t.iterations.map((x) => (
            <tr key={x.n} className={x.n === n ? 'cur' : ''} style={x.n > n ? { color: 'var(--ink-3)' } : undefined}>
              <td>{x.n}</td><td>{x.stride}</td><td>{x.active}</td><td>{x.activeWarps}</td>
              <td className={x.divergentWarps ? 't-fail' : ''}>{x.divergentWarps}</td><td>{x.slots}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p style={{ margin: '0.6rem 0 0' }} data-readout="totals" data-slots={t.slots} data-efficiency={t.efficiency.toFixed(4)}>
        So far: {sf.slots} slots consumed, {sf.results} results, efficiency {pct(sf.efficiency)}.{' '}
        <b>Whole loop: {t.slots} slots, {t.results} results, {pct(t.efficiency)}</b>; {t.divergentIterations} of {t.iterations.length} iterations
        have a divergent warp.
      </p>
    </div>
  );
}

type Show = InPlaceKind | 'both';
interface MapState {
  N: number;
  widthKey: string;
  show: Show;
  n: number;
}

function DivergenceLab() {
  const { reports } = useReports();
  const choices = useMemo(() => widthChoices(reports), [reports]);
  const [s, setS] = useState<MapState>({ N: 256, widthKey: 'book', show: 'both', n: 1 });
  const chosen = choices.find((c) => c.key === s.widthKey) ?? choices[0];
  const width = chosen.width;
  const traces = useMemo(
    () => ({ simple: iterations('simple', s.N, width), convergent: iterations('convergent', s.N, width) }),
    [s.N, width],
  );
  const L = traces.simple.iterations.length;
  const n = Math.min(s.n, L);
  const kinds: InPlaceKind[] = s.show === 'both' ? ['simple', 'convergent'] : [s.show];
  const book = { s: iterations('simple', 256, 32), c: iterations('convergent', 256, 32) };

  // Quiz me: local only, not recorded.
  const [quiz, setQuiz] = useState<{ kind: InPlaceKind; N: number; width: number; n: number } | null>(null);
  const [guess, setGuess] = useState('');
  const [revealed, setRevealed] = useState(false);
  const draw = () => {
    const kind: InPlaceKind = Math.random() < 0.5 ? 'simple' : 'convergent';
    const N = [256, 512, 1024, 2048][Math.floor(Math.random() * 4)];
    const w = Math.random() < 0.5 ? 32 : 64;
    const its = iterations(kind, N, w).iterations.length;
    setQuiz({ kind, N, width: w, n: 1 + Math.floor(Math.random() * its) });
    setGuess('');
    setRevealed(false);
  };
  const answer = quiz ? iterations(quiz.kind, quiz.N, quiz.width).iterations[quiz.n - 1].divergentWarps : 0;
  const reveal = () => {
    if (!quiz) return;
    setRevealed(true);
    setS({ N: quiz.N, widthKey: quiz.width === 32 ? 'book' : 'w64', show: quiz.kind, n: quiz.n });
  };

  return (
    <Bench
      title="Warp map"
      task="Same number of active threads in each iteration; count the warps that pay for them."
      foot={
        <>
          Each row is a warp, each cell a lane; filled lanes are active in this iteration, hatched lanes have no thread. A
          warp with any active lane costs all {width} of its lanes for the iteration, the book's accounting.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span style={FIELD}>
          N (values)
          <span className="seg" role="group" aria-label="N">
            {SIZES.map((N) => (
              <button key={N} aria-pressed={s.N === N} onClick={() => setS({ ...s, N, n: 1 })}>{N}</button>
            ))}
          </span>
        </span>
        <label className="field" style={{ maxWidth: '100%', minWidth: 0 }}>
          Warp width
          <select value={s.widthKey} onChange={(e) => setS({ ...s, widthKey: e.target.value, n: 1 })} aria-label="Warp width">
            {choices.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
        <span style={FIELD}>
          Kernel
          <span className="seg" role="group" aria-label="Kernel">
            {(['simple', 'convergent', 'both'] as Show[]).map((k) => (
              <button key={k} aria-pressed={s.show === k} onClick={() => setS({ ...s, show: k })}>{k === 'both' ? 'side by side' : k}</button>
            ))}
          </span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <button onClick={() => setS({ ...s, n: Math.max(1, n - 1) })} disabled={n <= 1}>Back</button>
        <button className="primary" onClick={() => setS({ ...s, n: Math.min(L, n + 1) })} disabled={n >= L}>Step</button>
        <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}>
          iteration
          <input type="range" min={1} max={L} value={n} onChange={(e) => setS({ ...s, n: Number(e.target.value) })} aria-label="Iteration" />
          <span data-readout="iteration-number">{n} of {L}</span>
        </label>
      </div>
      <div className="cols" style={{ gridTemplateColumns: kinds.length === 2 ? 'minmax(0, 1fr) minmax(0, 1fr)' : 'minmax(0, 1fr)' }}>
        {kinds.map((k) => <KernelPanel key={k} t={traces[k]} n={n} />)}
      </div>
      <p className="muted" style={{ margin: '0.9rem 0 0' }} data-readout="erratum">
        For 256 values and 32-wide warps the book prints, on p. 234, {BOOK_PRINTED.simpleSlots} slots for the simple kernel and
        divergent iterations falling "from {BOOK_PRINTED.divergentBefore === 10 ? 'ten' : BOOK_PRINTED.divergentBefore} to five". The
        book's own formula on p. 231 gives {book.s.slots}, and counting from the code gives {book.s.divergentIterations} divergent
        iterations against {book.c.divergentIterations}: {book.s.slots} slots to {book.c.slots}.
      </p>

      <div className="panel" style={{ marginTop: '1rem' }} data-part="quiz">
        <div className="mx-label">Quiz me</div>
        {!quiz ? (
          <p style={{ margin: '0 0 0.6rem' }}>Draw a kernel, a size, a width and an iteration, then count the divergent warps before the map shows them. Not recorded.</p>
        ) : (
          <p style={{ margin: '0 0 0.6rem' }} data-readout="quiz">
            The <b>{quiz.kind}</b> kernel on <b>{quiz.N}</b> values ({quiz.N / 2} threads), <b>{quiz.width}</b>-wide warps,
            iteration <b>{quiz.n}</b>. How many warps diverge?
          </p>
        )}
        <div className="row">
          <button onClick={draw}>{quiz ? 'Another' : 'Draw a question'}</button>
          {quiz ? (
            <>
              <input type="number" value={guess} onChange={(e) => { setGuess(e.target.value); setRevealed(false); }} aria-label="divergent warps" />
              <button className="primary" onClick={reveal} disabled={guess.trim() === ''}>Reveal</button>
            </>
          ) : null}
        </div>
        {quiz && revealed ? (
          <p style={{ margin: '0.6rem 0 0' }} data-readout="quiz-answer" data-answer={answer}>
            {Number(guess) === answer ? <span className="chip ok">right</span> : <span className="chip fail">not quite</span>}{' '}
            {answer} warp{answer === 1 ? '' : 's'} diverge{answer === 1 ? 's' : ''}: stride{' '}
            {iterations(quiz.kind, quiz.N, quiz.width).iterations[quiz.n - 1].stride}, and the map above now shows that iteration.
          </p>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- mirror it (exercise 3)
const OWNED_KEYS: OwnedKey[] = ['t', '2t', 't+B', '2t+1'];
const TEST_KEYS: TestKey[] = ['lt', 'mod', 'ge', 'mirror'];
const PARTNER_KEYS: PartnerKey[] = ['plus', 'minus'];
const WRITER_KEYS: WriterKey[] = ['first', 'middle', 'last'];

function specCode(s: Spec): string {
  return [
    '__global__ void MyKernel(float* input, float* output) {',
    `    unsigned int i = ${OWNED[s.owned].code};`,
    '    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {',
    `        if (${TESTS[s.test].code}) {`,
    `            input[i] += ${PARTNERS[s.partner].code};`,
    '        }',
    '        __syncthreads();',
    '    }',
    `    ${WRITERS[s.writer].code}`,
    '}',
  ].join('\n');
}

/** The input and the array after each iteration of a spec run, with the additions as arrows. */
function SpecGrid({ run, values }: { run: SpecRun; values: readonly number[] }) {
  const N = values.length;
  const CW = N > 8 ? 34 : 46;
  const CH = 24;
  const GAP = 24;
  const LW = 74;
  const y = (r: number) => 4 + r * (CH + GAP);
  const x = (c: number) => LW + c * CW;
  const rows = [{ stride: 0, after: [...values], ops: [] as SpecRun['iterations'][number]['ops'], conflicts: [] as number[] }, ...run.iterations];
  const W = LW + N * CW + 4;
  const H = y(rows.length - 1) + CH + 4;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="The input and the array after each iteration of the assembled kernel" data-part="spec-grid"
      style={{ width: W, maxWidth: '100%', minWidth: Math.min(W, 460), display: 'block' }}>
      <defs>
        <marker id="s104-arrow" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M0,0 L6,3 L0,6 z" fill="var(--ink-3)" />
        </marker>
      </defs>
      {rows.map((row, r) => {
        const written = new Set(row.ops.map((o) => o.own));
        const bad = new Set([...row.conflicts, ...row.ops.filter((o) => o.oob).map((o) => o.own)]);
        return (
          <g key={r} data-row={r}>
            <text x={LW - 8} y={y(r) + CH / 2 + 4} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
              {r === 0 ? 'input' : `stride ${row.stride}`}
            </text>
            {row.ops.map((o) => {
              const tx = x(o.own) + CW / 2;
              const inside = o.partner >= 0 && o.partner < N;
              return (
                <g key={o.t} opacity={0.8}>
                  <line x1={tx} y1={y(r - 1) + CH} x2={tx} y2={y(r) - 1} stroke="var(--ink-3)" markerEnd="url(#s104-arrow)" />
                  {inside ? (
                    <line x1={x(o.partner) + CW / 2} y1={y(r - 1) + CH} x2={tx + (o.partner > o.own ? 3 : -3)} y2={y(r) - 1}
                      stroke="var(--ink-3)" markerEnd="url(#s104-arrow)" />
                  ) : null}
                </g>
              );
            })}
            {row.after.map((v, c) => {
              const w = r === 0 || written.has(c);
              const b = bad.has(c);
              return (
                <g key={c} data-cell={c} data-value={Number.isNaN(v) ? '?' : v} data-written={w ? '1' : '0'} data-bad={b ? '1' : '0'}>
                  <rect x={x(c) + 1} y={y(r)} width={CW - 2} height={CH} rx="2"
                    fill={b ? 'var(--fail-wash)' : w ? 'var(--global-wash)' : 'transparent'}
                    stroke={b ? 'var(--fail)' : w ? 'var(--global)' : 'var(--rule)'} strokeDasharray={w ? undefined : '3 2'} />
                  <text x={x(c) + CW / 2} y={y(r) + CH / 2 + 4} fontSize="11" textAnchor="middle"
                    style={{ fill: b ? 'var(--fail)' : w ? 'var(--ink)' : 'var(--ink-3)', fontWeight: w && r > 0 ? 650 : 400 }}>
                    {Number.isNaN(v) ? '?' : v}
                  </text>
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

function MirrorIt() {
  const [spec, setSpec] = useState<Spec>(CONVERGENT_SPEC);
  const [data, setData] = useState<'fig' | 'ex6'>('fig');
  const [showRef, setShowRef] = useState(false);
  const ref = useLabFile('ch10/snippets/s104_mirrored.cu');
  const values = data === 'fig' ? FIG_VALUES : EXERCISE_6;
  const run = useMemo(() => runSpec(spec, values), [spec, values]);
  const N = values.length;
  const isMirror = (Object.keys(MIRROR_SPEC) as (keyof Spec)[]).every((k) => spec[k] === MIRROR_SPEC[k]);
  const pick = <K extends keyof Spec>(k: K, v: Spec[K]) => setSpec({ ...spec, [k]: v });
  const sel = <K extends keyof Spec>(k: K, label: string, keys: Spec[K][], code: (x: Spec[K]) => string) => (
    <label className="field" style={{ minWidth: 0, maxWidth: '100%' }}>
      {label}
      <select value={spec[k] as string} onChange={(e) => pick(k, e.target.value as Spec[K])} aria-label={label}>
        {keys.map((x) => <option key={x as string} value={x as string}>{code(x)}</option>)}
      </select>
    </label>
  );
  return (
    <Bench
      title="Mirror it"
      task="Exercise 3: change Fig. 10.8 so that threads own the second half and the total lands in the last element."
      foot={
        <>
          The loop is Fig. 10.8's, stride from blockDim.x down to 1, with its barrier; you choose the four parts. Each row
          shows what one iteration wrote, with arrows from the two elements an addition read. Red marks an element read in
          the same iteration that another thread writes it, or an index outside the input.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        {sel('owned', 'Owned element i =', OWNED_KEYS, (k) => OWNED[k].code)}
        {sel('test', 'Active when', TEST_KEYS, (k) => TESTS[k].code)}
        {sel('partner', 'Partner', PARTNER_KEYS, (k) => PARTNERS[k].code)}
        {sel('writer', 'After the loop', WRITER_KEYS, (k) => WRITERS[k].code)}
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Data
          <select value={data} onChange={(e) => setData(e.target.value as 'fig' | 'ex6')} aria-label="Data">
            <option value="fig">16 values (the exercise's figure)</option>
            <option value="ex6">exercise 6: 6, 2, 7, 4, 5, 8, 3, 1</option>
          </select>
        </label>
        <button onClick={() => setSpec(CONVERGENT_SPEC)}>Back to Fig. 10.8</button>
      </div>
      <div style={{ minWidth: 0 }}>
        <div style={{ overflowX: 'auto' }}>
          <SpecGrid run={run} values={values} />
        </div>
        <div style={{ marginTop: '0.7rem' }} data-readout="verdict" data-correct={run.correct ? '1' : '0'} data-last={run.totalInLast ? '1' : '0'}>
          {run.oob ? <p style={{ margin: '0 0 0.3rem' }}><span className="chip fail">out of bounds</span> {run.oob} read{run.oob > 1 ? 's' : ''} outside the input.</p> : null}
          {run.conflicts ? (
            <p style={{ margin: '0 0 0.3rem' }}>
              <span className="chip fail">conflict</span> {run.conflicts} element{run.conflicts > 1 ? 's' : ''} read in the same iteration that another thread
              writes {run.conflicts > 1 ? 'them' : 'it'}: the answer would depend on which thread goes first.
            </p>
          ) : null}
          <p style={{ margin: '0 0 0.3rem' }}>
            <span className="mono">*output</span> = {Number.isNaN(run.output) ? '?' : run.output}; the sum is {run.total}.{' '}
            {run.correct ? <span className="chip ok">right sum</span> : <span className="chip fail">wrong</span>}
          </p>
          <p style={{ margin: 0 }}>
            {run.totalInLast ? (
              <span className="chip ok">the total is in input[{N - 1}], the last element</span>
            ) : (
              <span className="muted">The last element, input[{N - 1}], ends at {Number.isNaN(run.iterations[run.iterations.length - 1].after[N - 1]) ? '?' : run.iterations[run.iterations.length - 1].after[N - 1]}, not the total.</span>
            )}
            {isMirror ? <span className="t-ok"> That is exercise 3's pattern.</span> : null}
          </p>
        </div>
      </div>
      <Code tabs={[{ label: 'your kernel', code: specCode(spec), tiers: TIERS }]} highlight={[2, 4, 5, 9]} />
      <div className="row" style={{ marginTop: '0.9rem' }}>
        <button onClick={() => setShowRef(!showRef)} aria-pressed={showRef}>{showRef ? 'Hide' : 'Show'} our answer</button>
      </div>
      {showRef && !missing(ref) ? (
        <Code
          tabs={[{ label: 's104_mirrored.cu', code: ref!, tiers: TIERS, status: STATUS.snippet }]}
          caption={
            <>
              Our answer, stored as <code>labs/ch10/snippets/s104_mirrored.cu</code>; the book prints none. A unit test ties
              the parts above to this file, and runs it on the course's CPU emulator.
            </>
          }
        />
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS104: Record<string, Question> = {
  'ch10/10.4/q-ex1': {
    kind: 'numeric',
    id: 'ch10/10.4/q-ex1',
    prompt: (
      <>
        Book exercise 1: the simple kernel (Fig. 10.5) on 1024 values, with 32-wide warps. How many warps diverge in the
        5th iteration?
      </>
    ),
    answer: 16,
    explain: (
      <p>
        512 threads make 16 warps. The 5th iteration has stride 16, so the active threads are the multiples of 16: lanes 0
        and 16 of every warp. Two active lanes out of 32 in each of the 16 warps: all 16 diverge.
      </p>
    ),
    source: 'ch. 10 §10.12, exercise 1, p. 249; answer ours',
  },
  'ch10/10.4/q-ex2': {
    kind: 'numeric',
    id: 'ch10/10.4/q-ex2',
    prompt: <>Book exercise 2: the convergent kernel (Fig. 10.8), same setup. How many warps diverge in the 5th iteration?</>,
    answer: 0,
    explain: (
      <p>
        The strides go 512, 256, 128, 64, 32. In the 5th iteration threads 0 to 31 are active: all of warp 0, and every
        other warp entirely idle. No warp diverges. Divergence starts in the 6th iteration, in warp 0 alone.
      </p>
    ),
    source: 'ch. 10 §10.12, exercise 2, p. 249; answer ours',
  },
  'ch10/10.4/q-eff-1024': {
    kind: 'numeric',
    id: 'ch10/10.4/q-eff-1024',
    prompt: <>The convergent kernel on 1024 values with 32-wide warps. What is its execution resource utilization efficiency, in percent?</>,
    answer: 89,
    tol: 0.02,
    unit: '%',
    explain: (
      <p>
        16 warps. While 32 or more threads are active, whole warps work: 16 + 8 + 4 + 2 + 1 = 31 warp-iterations. Then five
        iterations with one divergent warp. 36 × 32 = 1152 slots for 1023 additions: 89%. The efficiency rises with N,
        because the five divergent iterations at the end are a fixed cost.
      </p>
    ),
    source: 'ch. 10 §10.4, p. 234 (the formula); the 1024 case is ours',
  },
  'ch10/10.4/q-noncommutative': {
    kind: 'choice',
    id: 'ch10/10.4/q-noncommutative',
    prompt: <>For which operator could the convergent kernel give a different answer from the simple kernel, in exact arithmetic?</>,
    options: [
      { label: 'Integer max.', why: 'Max is commutative and associative: any order gives the same answer.' },
      { label: 'Integer sum (with no overflow).', why: 'Commutative and associative.' },
      { label: 'Integer product (with no overflow).', why: 'Commutative and associative.' },
      { label: 'String concatenation.', correct: true },
    ],
    explain: (
      <p>
        The simple kernel combines neighbours, so the operands stay in their original order and only the brackets move:
        associativity is enough. The convergent kernel combines element 0 with element 8, element 1 with element 9, so the
        operands themselves are reordered, and the operator must also be commutative. Concatenation is associative but not
        commutative: "ab" + "cd" is not "cd" + "ab". Floating-point addition is the everyday case where the order changes
        the last bits; section 10.2 showed it.
      </p>
    ),
    source: 'ch. 10 §10.2, p. 225; §10.4, pp. 232–233',
  },
  'ch10/10.4/q-w64-slots': {
    kind: 'numeric',
    id: 'ch10/10.4/q-w64-slots',
    prompt: (
      <>
        256 values (128 threads) on a device whose warps are 64 wide. How many thread slots does the convergent kernel
        consume, in the book's accounting?
      </>
    ),
    answer: 576,
    explain: (
      <p>
        Two warps of 64. Stride 128: both warps full. Stride 64: warp 0 full, warp 1 idle. Stride 32: warp 0 half active,
        so it diverges, one iteration earlier than with 32-wide warps. Then 16, 8, 4, 2, 1: warp 0 again. That is
        2 + 1 + 1 + 5 = 9 warp-iterations, × 64 = 576 slots, an efficiency of 255/576 = 44%. The simple kernel takes 960.
        The code never mentions the width; its efficiency depends on it all the same.
      </p>
    ),
    source: 'ours, with the book\'s accounting (p. 231); widths: notes/today/ch05-facts.md B5',
  },
  'ch10/10.4/q-ex6b': {
    kind: 'numeric',
    id: 'ch10/10.4/q-ex6b',
    prompt: (
      <>
        Book exercise 6b: run the convergent kernel on {'{'}6, 2, 7, 4, 5, 8, 3, 1{'}'}. What does <code>input[1]</code>{' '}
        hold at the end?
      </>
    ),
    answer: 15,
    explain: (
      <p>
        Strides 4, 2, 1. After stride 4: {'{'}11, 10, 10, 5, 5, 8, 3, 1{'}'}. After stride 2: {'{'}21, 15, 10, 5, ...{'}'}.
        After stride 1, only <code>input[0]</code> changes, to 36. Set the data of "Mirror it" to exercise 6 with Fig.
        10.8's parts to see every row. The answer is ours; the book prints none.
      </p>
    ),
    source: 'ch. 10 §10.12, exercise 6b, p. 250; answer ours',
  },
  'ch10/10.4/q-ex3': {
    kind: 'choice',
    id: 'ch10/10.4/q-ex3',
    prompt: (
      <>
        In the mirrored kernel of exercise 3, thread t owns <code>i = threadIdx.x + blockDim.x</code> and the stride halves
        from <code>blockDim.x</code> to 1. Which active test is right?
      </>
    ),
    options: [
      { label: <code>threadIdx.x &lt; stride</code>, why: 'That keeps the lowest threads active, which own the middle of the input, not its end.' },
      { label: <code>threadIdx.x &gt;= stride</code>, why: 'In the first iteration (stride = blockDim.x) no thread passes it.' },
      { label: <><code>threadIdx.x &gt;= blockDim.x - stride</code>, adding <code>input[i - stride]</code></>, correct: true },
      { label: <><code>threadIdx.x &gt;= blockDim.x - stride</code>, adding <code>input[i + stride]</code></>, why: 'i + stride runs past the end of the input.' },
    ],
    explain: (
      <p>
        The active threads must be the last <code>stride</code> ones, so that the work closes in on the last element, and
        each adds the element <code>stride</code> below its own. With 8 threads: stride 8 fills elements 8 to 15 from 0 to
        7; stride 4 fills 12 to 15 from 8 to 11; stride 2 fills 14 and 15; stride 1 fills 15. The last thread then writes{' '}
        <code>input[2*blockDim.x - 1]</code> to <code>*output</code>.
      </p>
    ),
    source: 'ch. 10 §10.12, exercise 3, p. 249; answer ours (labs/ch10/snippets/s104_mirrored.cu)',
  },
};

// ---------------------------------------------------------------- the page
export function S104() {
  const cu = useLabFile('ch10/shared/inplace_sum.cu');
  const metal = useLabFile('ch10/shared/inplace_sum.metal');
  const status = useLabStatus(LAB10_1);
  const simpleCu = !missing(cu) ? functionSource(cu!, 'SimpleSumReductionKernel') : null;
  const convCu = !missing(cu) ? functionSource(cu!, 'ConvergentSumReductionKernel') : null;
  const simpleMt = !missing(metal) ? functionSource(metal!, 'simple_sum') : null;
  const convMt = !missing(metal) ? functionSource(metal!, 'convergent_sum') : null;
  const cuDiff = simpleCu && convCu ? changedLines(simpleCu.text, convCu.text, 'SimpleSumReductionKernel', 'ConvergentSumReductionKernel') : [];
  const mtDiff = simpleMt && convMt ? changedLines(simpleMt.text, convMt.text, 'simple_sum', 'convergent_sum') : [];
  const c256 = iterations('convergent', 256, 32);
  return (
    <>
      <p>
        Look at which threads work in each iteration of the simple kernel. In the first, all of them. In the second, every
        other thread: half the lanes of every warp. By the sixth, one lane in 32. A warp with one active lane still
        occupies a full warp's worth of execution resources, and 31 of its lanes produce nothing. The book measures this
        with the <b>execution resource utilization efficiency</b>: the share of the resources consumed that produce a
        result.
      </p>
      <Source section="10.4" pages="230" />

      <h2>Counting the waste at 256 values</h2>
      <p>
        Take 256 values: 128 threads, 4 warps of 32. For the first six iterations every warp has at least one active
        thread, all with the same pattern. In the seventh only threads 0 and 64 work, so warps 0 and 2 run with one lane
        each and warps 1 and 3 are idle; in the eighth only warp 0 runs. Charge every active warp its 32 lanes and the loop
        consumes (4 × 6 + 2 + 1) × 32 = 864 thread slots. It produces 255 results, one per addition, so the efficiency is
        255/864, about 0.30.
      </p>
      <Source section="10.4" pages="230–231" />
      <BookTable />
      <p>
        The right-hand columns belong to the fix. It has exactly the same number of active threads in every iteration, and
        fewer than half the active warps.
      </p>

      <h2>Keep the active threads together</h2>
      <p>
        The trouble is that the partial sums, and so the threads that own them, drift further apart in every iteration.
        Turn it around. Let the threads own the first half of the input (thread t owns element t), start the stride at{' '}
        <code>blockDim.x</code> and halve it. In the first iteration, with 16 values, thread 0 adds element 8 into element
        0, thread 1 adds element 9 into element 1, and so on; in each later iteration the lowest half of the threads that
        were active stay active, each reaching as far as there are active threads. The active test becomes{' '}
        <code>threadIdx.x &lt; stride</code>. That is three changed lines.
      </p>
      <Source section="10.4" pages="231–233" figs="Figs. 10.7, 10.8" />
      {convCu ? (
        <Code
          tabs={[
            { label: 'CUDA (lab 10.1)', code: convCu.text, tiers: TIERS, status: status.cu, highlight: cuDiff },
            ...(convMt ? [{ label: 'Metal (lab 10.1)', code: convMt.text, tiers: TIERS, status: status.metal, highlight: mtDiff }] : []),
          ]}
          caption={
            <>
              <code>ConvergentSumReductionKernel</code> from <code>labs/ch10/shared/inplace_sum.cu</code> (and{' '}
              <code>convergent_sum</code> from <code>inplace_sum.metal</code>). The marked lines are the ones that differ
              from the simple kernel in section 10.3; everything else, including lab 10.1's row offset, is unchanged.
            </>
          }
        />
      ) : (
        <p className="muted">The kernel is loaded from <code>labs/ch10/shared/inplace_sum.cu</code>.</p>
      )}
      <p>
        This reorders more than the brackets. The tree now combines element 0 with element 8 and element 1 with element 9,
        so the operands meet in the order x0, x8, x1, x9, and so on. Section 10.2's warning comes due: the operator must be
        commutative as well as associative.
      </p>
      <Source section="10.4" pages="232–233" />
      <Ask q={questionsS104['ch10/10.4/q-noncommutative']} />

      <h2>Count it</h2>
      <Predict
        id="ch10/10.4/p-efficiency"
        prompt={
          <>
            Before you step the convergent kernel below: on 256 values with 32-wide warps, what efficiency do you expect, in
            percent? (The simple kernel's is 30%.)
          </>
        }
        unit="%"
      >
        {(v) => (
          <p data-readout="p-efficiency">
            The model counts {c256.slots} slots for {c256.results} results: {pct(c256.efficiency)}.{' '}
            {Math.abs(v - c256.efficiency * 100) <= 5
              ? 'That is what you expected.'
              : v < c256.efficiency * 100
                ? 'Higher than you expected: whole warps go idle together, so idle threads stop costing anything.'
                : 'Lower than you expected: the last five iterations still run one warp with a few active lanes.'}{' '}
            Now step it below and see where the slots go.
          </p>
        )}
      </Predict>
      <Simplification>
        <p>
          The map counts lanes, not time: a warp with any active lane is charged its full width for the iteration, which is
          the book's accounting. What a divergent iteration costs in time is for lab 10.1 to measure. Nothing here depends
          on data.
        </p>
      </Simplification>
      <DivergenceLab />

      <h2>Where the active threads are</h2>
      <p>
        Each iteration has as many active threads as before; what changed is where they are. In the second iteration
        threads 0 to 63 work and 64 to 127 do not, so warps 0 and 1 are fully active and warps 2 and 3 entirely idle. No
        warp diverges. That holds as long as at least 32 threads are active. In the last five iterations (16, 8, 4, 2 and 1
        active threads) the one remaining warp diverges. In total (4 + 2 + 1 + 5) × 32 = 384 slots, and 255/384 = 66%.
        Because the warps of a block take turns on hardware with limited resources, fewer slots also means less time.
      </p>
      <NoteGroup>
        <Source section="10.4" pages="233–234" />
        <Aside>
          <p>
            On p. 234 the book compares 384 with "736, the resources consumed by the kernel in Fig. 10.5", and says the
            divergent iterations drop "from ten to five". Its own formula on p. 231 gives 864, and the code diverges in seven of
            its eight iterations at this size. The warp map shows both counts.
          </p>
        </Aside>
      </NoteGroup>
      <p>
        The book's moral: a change of three lines in which thread does what, with a large effect, and you can only make it
        with confidence if you have the picture of how a warp executes.
      </p>
      <Source section="10.4" pages="234" />
      <Ask q={questionsS104['ch10/10.4/q-ex1']} label="Work it out, then check it on the map" />
      <Ask q={questionsS104['ch10/10.4/q-ex2']} label="Work it out" />
      <Ask q={questionsS104['ch10/10.4/q-eff-1024']} label="Work it out" />

      <h2>The width is not in the code</h2>
      <p>
        Neither kernel names 32, but both efficiencies depend on the warp width. On a device with 64-wide warps the
        convergent kernel diverges one iteration earlier: when 32 threads are active they fill only half of warp 0. The
        simple kernel loses as well. Set the width to 64 in the map and compare.
      </p>
      <Today checked="2026-11-20 (ch05-facts B5; ch04-facts C2)">
        <p>
          NVIDIA: 32. AMD: 64 on gfx9, 32 on gfx10 and later; do not assume either. Metal: undocumented, read from the
          pipeline.
        </p>
      </Today>
      <Ask q={questionsS104['ch10/10.4/q-w64-slots']} />
      <p>
        What a divergent iteration costs in time is a question for your device, not for this accounting. Chapter 4's{' '}
        <a href={href('ch04', 'labs')}>lab 4.2</a> measured divergence on its own; <a href={href('ch10', 'labs')}>lab 10.1</a>{' '}
        times the simple kernel against the convergent one.
      </p>
      <Beyond>
        <p>
          The width choices are the book's 32, the 64 AMD documents for gfx9, and any width your own CUDA or HIP device
          reported in <code>reports/runs</code>.
        </p>
      </Beyond>

      <h2>Mirror it</h2>
      <p>
        Book exercise 3 asks for the same idea turned around: the threads own the second half of the input, and the total
        ends in the last element. Start from Fig. 10.8's parts, which are selected below, and change them until the total
        lands in the last element with no conflict. The loop and the barrier stay as they are.
      </p>
      <NoteGroup>
        <Source section="10.12" pages="249" figs="exercise 3 and its figure" />
        <Simplification>
          <p>
            Each iteration is applied as if every active thread read before any wrote, and a value read in the same
            iteration that another thread writes is flagged as a conflict rather than raced. One block, no warps, no timing.
          </p>
        </Simplification>
      </NoteGroup>
      <MirrorIt />
      <Ask q={questionsS104['ch10/10.4/q-ex3']} />
      <Ask q={questionsS104['ch10/10.4/q-ex6b']} label="Work it out, then check it in Mirror it" />
    </>
  );
}
