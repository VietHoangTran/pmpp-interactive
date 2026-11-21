import { useMemo, useState, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { benchRows, num, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { SEGMENT_MODELS, type SegmentModelId } from '../../sim/coalesce';
import {
  CORANK_LINES,
  MAX_INPUT,
  PRESETS,
  coRankTrace,
  iRange,
  knownRange,
  mergeStable,
  parseSorted,
  partition,
  splitAt,
  type CoRankStep,
  type PresetId,
  type Tagged,
  type TestEval,
  type TieRule,
  type Verdict,
} from '../../sim/corank';
import {
  DATA,
  basicLayout,
  makeData,
  searchCost,
  threadsFor,
  warpStep,
  type DataId,
  type Formula,
  type Path,
} from '../../sim/mergeaccess';
import { CoRankScene } from './CoRankScene';
import { StagingPart, questionsStaging } from './StagingPart';

// ---------------------------------------------------------------- the code shown

// Copies of labs/ch13/snippets/merge_basic.cu, which ./verify.sh front-end-checks and
// labs/ch13/snippets/check.py runs on the emulator. snippets-s133.test.ts fails if a
// copy drifts from the file, and pins the line numbers this page refers to.
const MERGE_SEQ = `__device__ void merge_sequential(const int* A, int m, const int* B, int n, int* C) {
    int i = 0;  // index into A
    int j = 0;  // index into B
    int k = 0;  // index into C
    while ((i < m) && (j < n)) {  // both inputs have elements left
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    if (i == m) {  // A is used up: copy the rest of B
        while (j < n) {
            C[k++] = B[j++];
        }
    } else {  // B is used up: copy the rest of A
        while (i < m) {
            C[k++] = A[i++];
        }
    }
}`;

const CO_RANK = `__device__ int co_rank(int k, const int* A, int m, const int* B, int n) {
    int i = k < m ? k : m;                // i = min(k, m)
    int j = k - i;
    int i_low = 0 > (k - n) ? 0 : k - n;  // i_low = max(0, k - n)
    int j_low = 0 > (k - m) ? 0 : k - m;  // j_low = max(0, k - m)
    int delta;
    bool active = true;
    while (active) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            delta = ((i - i_low + 1) >> 1);  // ceil((i - i_low) / 2)
            j_low = j;
            j = j + delta;
            i = i - delta;
        } else if (j > 0 && i < m && B[j - 1] >= A[i]) {
            delta = ((j - j_low + 1) >> 1);  // ceil((j - j_low) / 2)
            i_low = i;
            i = i + delta;
            j = j - delta;
        } else {
            active = false;
        }
    }
    return i;
}`;

const MERGE_BASIC = `__global__ void merge_basic_kernel(const int* A, int m, const int* B, int n, int* C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int elementsPerThread = (m + n + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
    int k_curr = tid * elementsPerThread < m + n ? tid * elementsPerThread : m + n;
    int k_next = (tid + 1) * elementsPerThread < m + n ? (tid + 1) * elementsPerThread : m + n;
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr, &C[k_curr]);
}`;

/** The tie-rule bug of this page's question: the two comparisons exchanged (lines 9 and 14). */
const swapTieTests = (src: string) =>
  src.replace('A[i - 1] > B[j]', 'A[i - 1] >= B[j]').replace('B[j - 1] >= A[i]', 'B[j - 1] > A[i]');
const CO_RANK_SWAPPED = swapTieTests(CO_RANK);

/** Fig. 13.9 line 03 as printed (p. 313), quoted because the exact expression is the point. */
const LINE03_PRINTED = 'int elementsPerThread = ceil((m+n)/(blockDim.x*gridDim.x));';

const MERGE_TIERS: Record<string, Tier> = {
  A: 'global', B: 'global', C: 'global',
  i: 'reg', j: 'reg', k: 'reg', i_low: 'reg', j_low: 'reg', delta: 'reg', active: 'reg',
  tid: 'reg', elementsPerThread: 'reg', k_curr: 'reg', k_next: 'reg', i_curr: 'reg', i_next: 'reg', j_curr: 'reg', j_next: 'reg',
};
const SNIPPET_STATUS = 'emulator-checked and front-end-checked; unverified on CUDA and HIP';

// ---------------------------------------------------------------- small helpers

type Mark = 'fail' | 'ok';
const tagOf = (e: Tagged) => `${e.src}${e.idx}`;
const pct = (x: number) => `${(x * 100).toFixed(x === 1 || x === 0 ? 0 : 1)} %`;
const fmtInt = (x: number) => x.toLocaleString('en-US');
const VERDICT_TEXT: Record<Verdict, string> = { 'i-too-high': 'i is too high', 'j-too-high': 'j is too high', done: 'neither: done' };

/** The two tests' source text under a tie rule. */
const testCode = (rule: TieRule) => ({
  t1: `i > 0 && j < n && A[i−1] ${rule === 'book' ? '>' : '>='} B[j]`,
  t2: `j > 0 && i < m && B[j−1] ${rule === 'book' ? '>=' : '>'} A[i]`,
});

/** What one test found, in words, for a state (i, j) of arrays of sizes m and n. */
function testWords(which: 1 | 2, t: TestEval | null, i: number, j: number, m: number, n: number, rule: TieRule): string {
  if (!t) return 'not evaluated: the first test fired (else-if)';
  const op = which === 1 ? (rule === 'book' ? '>' : '≥') : rule === 'book' ? '≥' : '>';
  const notOp = which === 1 ? (rule === 'book' ? '≤' : '<') : rule === 'book' ? '<' : '≤';
  if (!t.inBounds) {
    if (which === 1) return i > 0 ? `j < n is false (j = n = ${n}), so nothing is read` : 'i > 0 is false, so nothing is read';
    return j > 0 ? `i < m is false (i = m = ${m}), so nothing is read` : 'j > 0 is false, so nothing is read';
  }
  const [l, r] = which === 1 ? [`A[${i - 1}]`, `B[${j}]`] : [`B[${j - 1}]`, `A[${i}]`];
  return `${l} = ${t.left} ${t.holds ? op : notOp} ${r} = ${t.right}`;
}

function verdictChip(t: TestEval | null) {
  if (!t) return <span className="chip">skipped</span>;
  return t.holds ? <span className="chip fail">fires</span> : <span className="chip ok">passes</span>;
}

// ---------------------------------------------------------------- drawing A, B and C

const VIEW_W = 720;
const NARROW_MIN = 560;

interface ArraysProps {
  A: readonly number[];
  B: readonly number[];
  bars?: { i: number; j: number; iLow?: number; jLow?: number };
  markA?: Map<number, Mark>;
  markB?: Map<number, Mark>;
  shadeA?: (p: number) => boolean;
  shadeB?: (p: number) => boolean;
  window?: [number, number];
  C: readonly Tagged[];
  k: number;
  markC?: Map<number, Mark>;
  shadeC?: (p: number) => boolean;
  kLabel: string;
  label: string;
}

function Arrays(pr: ArraysProps) {
  const m = pr.A.length;
  const n = pr.B.length;
  const cw = Math.max(18, Math.min(44, Math.floor((VIEW_W - 90) / (m + n + 2))));
  const top = 42;
  const ax0 = 34;
  const bx0 = ax0 + Math.max(m, 1) * cw + 2 * cw;
  const cy = top + cw + 74;
  const cx0 = Math.max(34, (VIEW_W - (m + n) * cw) / 2);
  const font = Math.max(9.5, Math.min(14, cw * 0.36));
  const H = cy + cw + 36;

  const row = (x0: number, y: number, name: string, vals: readonly number[], mark?: Map<number, Mark>, shade?: (p: number) => boolean, tags?: readonly Tagged[]) => (
    <g>
      <text x={x0 - 8} y={y + cw / 2 + 5} fontSize="13" fontWeight="650" textAnchor="end" style={{ fill: 'var(--global)' }}>{name}</text>
      {vals.length === 0 ? (
        <g>
          <rect x={x0} y={y} width={cw - 2} height={cw - 2} rx="2" fill="none" stroke="var(--rule)" strokeDasharray="3 3" />
          <text x={x0 + cw + 4} y={y + cw / 2 + 4} fontSize="11" style={{ fill: 'var(--ink-3)' }}>empty</text>
        </g>
      ) : null}
      {vals.map((v, p) => {
        const mk = mark?.get(p);
        const sh = !mk && shade?.(p);
        return (
          <g key={p} data-cell={`${name}${p}`} data-mark={mk ?? (sh ? 'shade' : '')}>
            <rect x={x0 + p * cw} y={y} width={cw - 2} height={cw - 2} rx="2"
              fill={mk ? `var(--${mk}-wash)` : sh ? 'var(--global-wash)' : 'var(--cell)'}
              stroke={mk ? `var(--${mk})` : sh ? 'var(--global)' : 'var(--rule)'} strokeWidth={mk ? 2 : 1} />
            <text x={x0 + p * cw + (cw - 2) / 2} y={y + cw / 2 + font / 3} fontSize={font} textAnchor="middle"
              style={{ fill: mk ? `var(--${mk})` : 'var(--ink)', fontWeight: mk ? 700 : 450 }}>{v}</text>
            {tags ? (
              <text x={x0 + p * cw + (cw - 2) / 2} y={y + cw + 9} fontSize={Math.min(10, font * 0.8)} textAnchor="middle"
                style={{ fill: mk === 'fail' ? 'var(--fail)' : 'var(--ink-2)' }}>{tagOf(tags[p])}</text>
            ) : (
              <text x={x0 + p * cw + (cw - 2) / 2} y={y - 3} fontSize="8.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{p}</text>
            )}
          </g>
        );
      })}
    </g>
  );
  const bar = (x: number, y: number, text: string, low: boolean, nudge = 0) => (
    <g>
      <line x1={x - 1 + nudge} y1={low ? y + 4 : y - 13} x2={x - 1 + nudge} y2={low ? y + cw + 6 : y + cw - 6} stroke="var(--ink)"
        strokeWidth={low ? 1.5 : 2.8} strokeDasharray={low ? '4 3' : undefined} />
      <text x={x - 1 + nudge} y={low ? y + cw + 17 : y - 16} fontSize="11" fontWeight={low ? 450 : 650} textAnchor="middle">{text}</text>
    </g>
  );
  const b = pr.bars;
  return (
    <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${VIEW_W} ${H}`} role="img" aria-label={pr.label} style={{ width: '100%', minWidth: NARROW_MIN, display: 'block' }}>
        {row(ax0, top, 'A', pr.A, pr.markA, pr.shadeA)}
        {row(bx0, top, 'B', pr.B, pr.markB, pr.shadeB)}
        {b ? (
          <g>
            {bar(ax0 + b.i * cw, top, `i = ${b.i}`, false)}
            {b.iLow != null ? bar(ax0 + b.iLow * cw, top, `i_low = ${b.iLow}`, true, b.iLow === b.i ? 3 : 0) : null}
            {bar(bx0 + b.j * cw, top, `j = ${b.j}`, false)}
            {b.jLow != null ? bar(bx0 + b.jLow * cw, top, `j_low = ${b.jLow}`, true, b.jLow === b.j ? 3 : 0) : null}
          </g>
        ) : null}
        {pr.window ? (
          <g data-window={`${pr.window[0]}-${pr.window[1]}`}>
            <line x1={ax0 + pr.window[0] * cw - 1} y1={top + cw + 30} x2={ax0 + pr.window[1] * cw - 1} y2={top + cw + 30} stroke="var(--ink-2)" strokeWidth="2" />
            {Array.from({ length: Math.max(0, pr.window[1] - pr.window[0] + 1) }, (_, q) => (
              <circle key={q} cx={ax0 + (pr.window![0] + q) * cw - 1} cy={top + cw + 30} r="3.5" fill="var(--ink)" />
            ))}
            <text x={ax0 + pr.window[1] * cw + 8} y={top + cw + 34} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
              {pr.window[1] - pr.window[0] + 1} candidate{pr.window[0] === pr.window[1] ? '' : 's'} left
            </text>
          </g>
        ) : null}
        {row(cx0, cy, 'C', pr.C.map((e) => e.key), pr.markC, pr.shadeC, pr.C)}
        <line x1={cx0 + pr.k * cw - 1} y1={cy - 12} x2={cx0 + pr.k * cw - 1} y2={cy + cw + 12} stroke="var(--ink)" strokeWidth="2.8" />
        <text x={cx0 + pr.k * cw - 1} y={cy - 16} fontSize="11" fontWeight="650" textAnchor="middle">{pr.kLabel}</text>
      </svg>
    </div>
  );
}

// ---------------------------------------------------------------- the co-rank explorer

const RUNNING = PRESETS[0].make(0);

function CoRankExplorer() {
  const [arr, setArr] = useState<{ A: number[]; B: number[]; label: string }>({ ...RUNNING, label: PRESETS[0].label });
  const [seed, setSeed] = useState(1);
  const [textA, setTextA] = useState(RUNNING.A.join(', '));
  const [textB, setTextB] = useState(RUNNING.B.join(', '));
  const [err, setErr] = useState<string | null>(null);
  const [k, setK] = useState(3);
  const [mode, setMode] = useState<'search' | 'split'>('search');
  const [rule, setRule] = useState<TieRule>('book');
  const [pos, setPos] = useState(0);
  const [predict, setPredict] = useState(false);
  const [feedback, setFeedback] = useState<{ ok: boolean; text: string } | null>(null);
  const [force, setForce] = useState<number | null>(null);

  const { A, B } = arr;
  const m = A.length;
  const n = B.length;
  const kk = Math.min(k, m + n);
  const trace = useMemo(() => coRankTrace(kk, A, B, rule), [kk, A, B, rule]);
  const S = trace.steps.length;
  const at = Math.min(pos, 2 * S - 1);
  const it = Math.floor(at / 2);
  const shown = at % 2 === 1;
  const st: CoRankStep = trace.steps[it];
  const C = useMemo(() => mergeStable(A, B), [A, B]);
  const [lo, hi] = iRange(kk, m, n);
  const fi = force == null ? hi : Math.max(lo, Math.min(hi, force));
  const split = useMemo(() => splitAt(kk, fi, A, B), [kk, fi, A, B]);
  const moves = trace.steps.slice(0, it + (shown ? 1 : 0)).filter((s) => s.test !== 'done').length;

  const reset = () => { setPos(0); setFeedback(null); };
  const load = (A2: number[], B2: number[], label: string, k2?: number) => {
    setArr({ A: A2, B: B2, label });
    setTextA(A2.join(', '));
    setTextB(B2.join(', '));
    setErr(null);
    setK(k2 ?? Math.min(k, A2.length + B2.length));
    setForce(null);
    reset();
  };
  const pick = (id: PresetId) => {
    const p = PRESETS.find((x) => x.id === id)!;
    const s = id === 'random' || id === 'duplicates' ? seed + 1 : seed;
    if (s !== seed) setSeed(s);
    const d = p.make(s);
    load(d.A, d.B, id === 'random' || id === 'duplicates' ? `${p.label}, seed ${s}` : p.label);
  };
  const useOwn = () => {
    const a = parseSorted(textA);
    const b = parseSorted(textB);
    if (!a.ok) return setErr(`A: ${a.error}`);
    if (!b.ok) return setErr(`B: ${b.error}`);
    load(a.values, b.values, 'your arrays');
  };

  const choose = (v: Verdict) => {
    const ok = v === st.test;
    setFeedback({ ok, text: `${ok ? 'Right' : `Not quite: you said ${VERDICT_TEXT[v]}`}. ${because(st, rule)}` });
    setPos(at + 1);
  };

  // What to light up.
  const marks = { a: new Map<number, Mark>(), b: new Map<number, Mark>() };
  if (mode === 'search' && shown) {
    const put = (t: TestEval | null, which: 1 | 2) => {
      if (!t?.inBounds) return;
      const mk: Mark = t.holds ? 'fail' : 'ok';
      if (which === 1) { marks.a.set(st.i - 1, mk); marks.b.set(st.j, mk); }
      else { marks.b.set(st.j - 1, mk); marks.a.set(st.i, mk); }
    };
    put(st.t1, 1);
    put(st.t2, 2);
  }
  const done = shown && st.test === 'done';
  const resultPrefix = done ? mergeStable(A.slice(0, st.i), B.slice(0, st.j)) : null;
  const prefixOk = resultPrefix ? resultPrefix.every((e, p) => e.src === C[p].src && e.idx === C[p].idx) : true;

  // Split mode: which elements of C are out of place.
  const splitC = [...split.prefix, ...split.suffix];
  const splitMarks = new Map<number, Mark>();
  if (!split.sortedKeys) {
    const lastKey = split.prefix[split.prefix.length - 1]?.key;
    const firstKey = split.suffix[0]?.key;
    split.prefix.forEach((e, p) => { if (e.key > firstKey) splitMarks.set(p, 'fail'); });
    split.suffix.forEach((e, p) => { if (e.key < lastKey) splitMarks.set(split.prefix.length + p, 'fail'); });
  } else if (!split.stable) {
    splitC.forEach((e, p) => { if (e.src !== C[p].src || e.idx !== C[p].idx) splitMarks.set(p, 'fail'); });
  }

  const lines = mode === 'search' ? (at === 0 ? CORANK_LINES.init : !shown ? [8] : CORANK_LINES[st.test]) : [];
  const codes = testCode(rule);

  return (
    <Bench
      title="Co-rank explorer"
      task="Predict each step before you take it. Then break it on purpose."
      foot={
        <>
          Runs Fig. 13.5 exactly (src/sim/corank.ts): the same tests, the same short-circuit reads, the same moves. Cells
          that a test reads are outlined: red where the test fires, green where it passes. Under C, each element's tag
          says which array and index it came from.
        </>
      }
    >
      <div
        data-explorer="corank"
        data-k={kk}
        data-mode={mode}
        data-rule={rule}
        data-iteration={it}
        data-shown={shown ? 1 : 0}
        data-i={st.i}
        data-j={st.j}
        data-ilow={st.iLow}
        data-jlow={st.jLow}
        data-verdict={shown ? st.test : ''}
        data-moves={moves}
        data-bound={trace.bound}
        data-result={trace.result}
        data-force-i={fi}
        data-valid={split.valid ? 1 : 0}
        data-failing={split.failing ?? ''}
        data-sorted={split.sortedKeys ? 1 : 0}
        data-stable={split.stable ? 1 : 0}
        data-arrays={`${A.join(',')}|${B.join(',')}`}
      >
        <div className="row" style={{ marginBottom: '0.6rem' }}>
          <span className="muted">Arrays:</span>
          {PRESETS.map((p) => (
            <button key={p.id} className="quiet" onClick={() => pick(p.id)}>{p.label}</button>
          ))}
        </div>
        <div className="row" style={{ marginBottom: '0.6rem', alignItems: 'flex-end' }}>
          <label className="field">
            your A (sorted, at most {MAX_INPUT})
            <input type="text" value={textA} onChange={(e) => setTextA(e.target.value)} aria-label="your A" style={{ width: '13rem', maxWidth: '100%' }} />
          </label>
          <label className="field">
            your B
            <input type="text" value={textB} onChange={(e) => setTextB(e.target.value)} aria-label="your B" style={{ width: '13rem', maxWidth: '100%' }} />
          </label>
          <button onClick={useOwn}>Use these</button>
          {err ? <span className="t-fail" role="alert">{err}</span> : <span className="muted">Now: {arr.label} (m = {m}, n = {n}).</span>}
        </div>
        <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
          <label className="field">
            k = {kk} (split C[0..{kk}) between A and B)
            <input type="range" min={0} max={m + n} value={kk} aria-label="k"
              onChange={(e) => { setK(Number(e.target.value)); setForce(null); reset(); }} />
          </label>
          <span className="seg" role="group" aria-label="Mode">
            <button aria-pressed={mode === 'search'} onClick={() => setMode('search')}>Step the search</button>
            <button aria-pressed={mode === 'split'} onClick={() => setMode('split')}>Try any split</button>
          </span>
        </div>

        {mode === 'search' ? (
          <>
            <div className="row" style={{ marginBottom: '0.7rem' }}>
              <button onClick={reset} disabled={at === 0}>Restart</button>
              <button onClick={() => { setPos(at - 1); setFeedback(null); }} disabled={at === 0}>Back</button>
              {predict && !shown ? (
                <span className="row" style={{ gap: '0.4rem' }} role="group" aria-label="Your prediction">
                  <span className="muted">Which test fires?</span>
                  <button onClick={() => choose('i-too-high')}>i too high</button>
                  <button onClick={() => choose('j-too-high')}>j too high</button>
                  <button onClick={() => choose('done')}>done</button>
                </span>
              ) : (
                <button className="primary" onClick={() => { setPos(at + 1); setFeedback(null); }} disabled={at >= 2 * S - 1}>Step</button>
              )}
              <button onClick={() => { setPos(2 * S - 1); setFeedback(null); }} disabled={at >= 2 * S - 1}>Run to end</button>
              <label className="check"><input type="checkbox" checked={predict} onChange={(e) => setPredict(e.target.checked)} /> predict mode</label>
              <label className="check">
                <input type="checkbox" checked={rule === 'swapped'} onChange={(e) => { setRule(e.target.checked ? 'swapped' : 'book'); reset(); }} /> swap the tie tests
              </label>
            </div>
            <div className={'narrate' + (feedback && !feedback.ok ? ' bad' : '')} role="status" style={{ marginBottom: '0.8rem' }}>
              {feedback ? <><b className={feedback.ok ? 't-ok' : 't-fail'}>{feedback.text}</b><br /></> : null}
              {narrate(trace, at, m, n, rule)}
            </div>
          </>
        ) : (
          <>
            <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
              <label className="field">
                force i = {fi}, so j = {kk - fi} (i can be {lo} to {hi})
                <input type="range" min={lo} max={hi} value={fi} aria-label="force i" disabled={lo === hi} onChange={(e) => setForce(Number(e.target.value))} />
              </label>
            </div>
            <div className={'narrate' + (split.valid ? '' : ' bad')} role="status" style={{ marginBottom: '0.8rem' }}>
              {splitWords(split)}
            </div>
          </>
        )}

        {mode === 'search' ? (
          <Arrays
            label="A, B and C with the search's marker bars"
            A={A} B={B}
            bars={{ i: st.i, j: st.j, iLow: st.iLow, jLow: st.jLow }}
            markA={marks.a} markB={marks.b}
            window={rule === 'book' ? knownRange(trace, shown ? it : it - 1) : undefined}
            C={C} k={kk} kLabel={`k = ${kk}`}
            shadeC={(p) => p < kk}
          />
        ) : (
          <Arrays
            label="A forced split: what each of two threads would write"
            A={A} B={B}
            bars={{ i: fi, j: kk - fi }}
            shadeA={(p) => p < fi} shadeB={(p) => p < kk - fi}
            C={splitC} k={kk} kLabel={`thread 0 | thread 1 at k = ${kk}`}
            markC={splitMarks} shadeC={(p) => p < kk}
          />
        )}

        {mode === 'search' ? (
          <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 5fr) minmax(0, 6fr)', marginTop: '0.6rem' }}>
            <div>
              <table className="data" style={{ margin: 0 }} aria-label="The two tests">
                <thead>
                  <tr><th>Test</th><th>Condition</th><th>Here</th><th /></tr>
                </thead>
                <tbody>
                  <tr data-test="1">
                    <td>i too high?</td>
                    <td><code>{codes.t1}</code></td>
                    <td>{shown ? testWords(1, st.t1, st.i, st.j, m, n, rule) : '?'}</td>
                    <td>{shown ? verdictChip(st.t1) : null}</td>
                  </tr>
                  <tr data-test="2">
                    <td>j too high?</td>
                    <td><code>{codes.t2}</code></td>
                    <td>{shown ? testWords(2, st.t2, st.i, st.j, m, n, rule) : '?'}</td>
                    <td>{shown ? verdictChip(st.t2) : null}</td>
                  </tr>
                </tbody>
              </table>
              <p className="row" style={{ margin: '0.7rem 0 0' }}>
                <span className="chip ok" data-sum>i + j = {st.i} + {st.j} = {st.i + st.j} = k</span>
                <span className="chip" data-moves-chip>
                  moves {moves} of at most {trace.bound} (⌈log2(w + 1)⌉, w = {trace.candidates - 1})
                </span>
                {done ? (
                  <span className={'chip ' + (prefixOk ? 'ok' : 'fail')} data-prefix-ok={prefixOk ? 1 : 0}>
                    {prefixOk ? `C[0..${kk}) = A[0..${st.i}) + B[0..${st.j}), as the stable merge has it` : `the stable merge takes ${C.slice(0, kk).filter((e) => e.src === 'A').length} from A, not ${st.i}`}
                  </span>
                ) : null}
              </p>
            </div>
            <Code
              tabs={[
                rule === 'book'
                  ? { label: 'co_rank', code: CO_RANK, tiers: MERGE_TIERS, status: SNIPPET_STATUS }
                  : { label: 'co_rank, tie tests swapped', code: CO_RANK_SWAPPED, tiers: MERGE_TIERS, status: 'unverified: a deliberate defect' },
              ]}
              highlight={lines}
            />
          </div>
        ) : null}
      </div>
    </Bench>
  );
}

/** Why a step's verdict is what it is, in one sentence. */
function because(s: CoRankStep, rule: TieRule): string {
  const g = rule === 'book' ? ['>', '≥'] : ['≥', '>'];
  if (s.test === 'i-too-high') return `A[${s.i - 1}] = ${s.t1.left} ${g[0]} B[${s.j}] = ${s.t1.right}, so i is too high.`;
  if (s.test === 'j-too-high') return `B[${s.j - 1}] = ${s.t2!.left} ${g[1]} A[${s.i}] = ${s.t2!.right}, so j is too high.`;
  return 'Neither test fires, so the loop stops.';
}

function narrate(trace: ReturnType<typeof coRankTrace>, at: number, m: number, n: number, rule: TieRule): ReactNode {
  const k = trace.k;
  const it = Math.floor(at / 2);
  const st = trace.steps[it];
  if (at === 0) {
    return (
      <>
        Start: i = min(k, m) = min({k}, {m}) = {st.i}; j = k − i = {st.j}; i_low = max(0, k − n) = max(0, {k - n}) = {st.iLow};
        j_low = max(0, k − m) = max(0, {k - m}) = {st.jLow}. Press Step to run the two tests.
      </>
    );
  }
  if (at % 2 === 0) return <>Iteration {it}: i = {st.i}, j = {st.j}, i_low = {st.iLow}, j_low = {st.jLow}. Which test fires?</>;
  if (st.test === 'i-too-high') {
    const d = st.delta!;
    return (
      <>
        Iteration {it}: {because(st, rule)} delta = ⌈(i − i_low) / 2⌉ = ⌈({st.i} − {st.iLow}) / 2⌉ = {d}; j_low = j = {st.j}; j = {st.j} + {d} = {st.j + d}; i = {st.i} − {d} = {st.i - d}.
      </>
    );
  }
  if (st.test === 'j-too-high') {
    const d = st.delta!;
    return (
      <>
        Iteration {it}: the first test does not fire. {because(st, rule)} delta = ⌈(j − j_low) / 2⌉ = ⌈({st.j} − {st.jLow}) / 2⌉ = {d}; i_low = i = {st.i}; i = {st.i} + {d} = {st.i + d}; j = {st.j} − {d} = {st.j - d}.
      </>
    );
  }
  return (
    <>
      Iteration {it}: neither test fires. co_rank({k}) returns i = {st.i}, so j = {st.j}: the first {k} outputs are A[0..{st.i}) and
      B[0..{st.j}). {trace.moves} move{trace.moves === 1 ? '' : 's'}, against a bound of {trace.bound} for {trace.candidates} candidate
      {trace.candidates === 1 ? '' : 's'}.
    </>
  );
}

function splitWords(s: ReturnType<typeof splitAt>): ReactNode {
  const share = `Thread 0 merges A[0..${s.i}) and B[0..${s.j}) into C[0..${s.k}); thread 1 merges the rest.`;
  if (s.valid) return <>{share} Both tests pass: i = {s.i} is the co-rank of {s.k}, and C comes out exactly as the stable merge.</>;
  const fired =
    s.failing === 'i-too-high'
      ? `The first test fires: A[${s.i - 1}] = ${s.t1.left} > B[${s.j}] = ${s.t1.right}, so i is too high. Move i down.`
      : `The second test fires: B[${s.j - 1}] = ${s.t2!.left} ≥ A[${s.i}] = ${s.t2!.right}, so j is too high. Move i up.`;
  const result = !s.sortedKeys
    ? 'C is not sorted at the cut between the threads.'
    : 'The keys happen to be in order, but equal keys are out of their stable order: the tie went the wrong way.';
  return <>{share} <b className="t-fail">{result}</b> {fired}</>;
}

// ---------------------------------------------------------------- the worked example of p. 314

function WorkedExample() {
  const slices = partition(RUNNING.A, RUNNING.B, 3, 3);
  return (
    <table className="data wide" style={{ maxWidth: '50rem' }} aria-label="The basic kernel on the running example, three threads" data-worked>
      <thead>
        <tr><th>thread</th><th className="num">k_curr</th><th className="num">k_next</th><th className="num">i_curr</th><th className="num">i_next</th><th className="num">j_curr</th><th className="num">j_next</th><th>merge_sequential call</th></tr>
      </thead>
      <tbody>
        {slices.map((s) => (
          <tr key={s.t}>
            <td>{s.t}</td>
            <td className="num">{s.kCurr}</td><td className="num">{s.kNext}</td>
            <td className="num">{s.iCurr}</td><td className="num">{s.iNext}</td>
            <td className="num">{s.jCurr}</td><td className="num">{s.jNext}</td>
            <td><code>(&amp;A[{s.iCurr}], {s.iNext - s.iCurr}, &amp;B[{s.jCurr}], {s.jNext - s.jCurr}, &amp;C[{s.kCurr}])</code></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------- one warp of the basic kernel

/** Warp widths the reader's devices reported: lab 6.1's simd_width row, else the device's warpSize. */
function useDeviceWidths(): { backend: string; device: string; width: number }[] {
  const { reports } = useReports();
  return realBackends(reports)
    .filter((b) => b !== 'race' && !/emulator/i.test(String(reports[b].device?.name ?? '')))
    .flatMap((b) => {
      const row = benchRows(reports[b], 'ch06/lab1_coalescing').find((r) => r.name === 'simd_width');
      const w = num(row?.width) ?? num(Number(reports[b].device.warpSize));
      return w ? [{ backend: b, device: String(reports[b].device.name ?? b), width: w }] : [];
    });
}

const PATH_LETTER: Record<Path, string> = { A: 'A', B: 'B', tailA: 'a', tailB: 'b', idle: '·' };
const PATH_WORDS: Record<Exclude<Path, 'idle'>, string> = {
  A: 'main loop, takes A[i]',
  B: 'main loop, takes B[j]',
  tailA: 'copies the rest of A',
  tailB: 'copies the rest of B',
};

function Strips({ layout, warp, warpWidth, s }: { layout: ReturnType<typeof basicLayout>; warp: number; warpWidth: number; s: number }) {
  const W = 700;
  const L = 60;
  const span = W - L - 64;
  const m = layout.A.length;
  const n = layout.B.length;
  const N = layout.total;
  const strips = [
    { name: 'A', len: m, y: 10, lo: (t: number) => layout.work[t].iCurr, hi: (t: number) => layout.work[t].iNext },
    { name: 'B', len: n, y: 48, lo: (t: number) => layout.work[t].jCurr, hi: (t: number) => layout.work[t].jNext },
    { name: 'C', len: N, y: 86, lo: (t: number) => layout.work[t].kCurr, hi: (t: number) => layout.work[t].kNext },
  ];
  const first = warp * warpWidth;
  const last = Math.min(layout.threads, first + warpWidth) - 1;
  const ws = warpStep(layout, warp, warpWidth, s);
  const unw = layout.unwritten;
  return (
    <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${W} 122`} role="img" aria-label="Each thread's slices of A, B and C" style={{ width: '100%', minWidth: NARROW_MIN, display: 'block' }}>
        {strips.map((st) => {
          const sc = (x: number) => L + (st.len ? (x / st.len) * span : 0);
          return (
            <g key={st.name}>
              <text x={L - 8} y={st.y + 14} fontSize="11.5" fontWeight="650" textAnchor="end" style={{ fill: 'var(--global)' }}>{st.name}</text>
              <rect x={L} y={st.y} width={span} height={20} fill="var(--cell)" stroke="var(--rule)" />
              {/* each thread's slice, shaded */}
              {layout.work.map((w) => {
                const a = st.lo(w.t);
                const b = st.hi(w.t);
                if (b <= a) return null;
                const mine = w.t >= first && w.t <= last;
                // Alternate shades so neighbouring slices can be told apart; this warp's threads in the global tint.
                return (
                  <rect key={w.t} x={sc(a)} y={st.y + 1} width={Math.max(0.6, sc(b) - sc(a))} height={18}
                    fill={mine ? 'var(--global-wash)' : 'var(--rule-soft)'} opacity={w.t % 2 ? 0.45 : 1} />
                );
              })}
              {first < layout.threads ? (
                <rect x={sc(st.lo(first))} y={st.y} width={Math.max(0.6, sc(st.hi(last)) - sc(st.lo(first)))} height={20}
                  fill="none" stroke="var(--global)" strokeWidth="1.2" />
              ) : null}
              {st.name === 'C' && unw.length ? (
                <rect x={sc(unw[0])} y={st.y} width={sc(N) - sc(unw[0])} height={20} fill="var(--fail-wash)" stroke="var(--fail)" data-unwritten-band />
              ) : null}
              {ws.lanes.flatMap((ln) =>
                st.name === 'C'
                  ? ln.write == null ? [] : [<line key={'w' + ln.lane} x1={sc(ln.write + 0.5)} y1={st.y - 5} x2={sc(ln.write + 0.5)} y2={st.y + 25} stroke="var(--ink)" strokeWidth="1.4" />]
                  : ln.reads.filter((r) => r.arr === st.name).map((r) => (
                      <line key={'r' + ln.lane + r.index} x1={sc(r.index + 0.5)} y1={st.y - 5} x2={sc(r.index + 0.5)} y2={st.y + 25} stroke="var(--ink)" strokeWidth="1.4" />
                    )),
              )}
              <text x={L + span + 8} y={st.y + 14} fontSize="10" style={{ fill: 'var(--ink-3)' }}>{fmtInt(st.len)}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function WarpExplorer() {
  const [data, setData] = useState<DataId>('distinct');
  const [total, setTotal] = useState(256);
  const [per, setPer] = useState(8);
  const [ww, setWw] = useState(32);
  const [model, setModel] = useState<SegmentModelId>('nvidia');
  const [custom, setCustom] = useState(32);
  const [formula, setFormula] = useState<Formula>('corrected');
  const [warp, setWarp] = useState(0);
  const [s, setS] = useState(0);
  const widths = useDeviceWidths();

  const seg = SEGMENT_MODELS[model].segmentBytes ?? custom;
  const d = useMemo(() => makeData(data, total, 1), [data, total]);
  const threads = threadsFor(total, per);
  const layout = useMemo(() => basicLayout(d.A, d.B, threads, formula), [d, threads, formula]);
  const E = layout.perThread;
  const warps = Math.ceil(threads / ww);
  const w = Math.min(warp, warps - 1);
  const step = Math.min(s, Math.max(0, E - 1));
  const ws = warpStep(layout, w, ww, step, 4, seg);
  const cost = searchCost(layout);
  const lanesA = ws.lanes.filter((l) => l.reads.some((r) => r.arr === 'A')).length;
  const lanesB = ws.lanes.filter((l) => l.reads.some((r) => r.arr === 'B')).length;
  const lanesC = ws.lanes.filter((l) => l.write != null).length;
  // Distinct loops among the paths: the main loop (A or B) and the two tail loops.
  const loops = new Set(ws.paths.map((p) => (p === 'A' || p === 'B' ? 'main' : p))).size;
  const unit = SEGMENT_MODELS[model].unit;
  const effChip = (e: number, used: number) =>
    used === 0 ? <span className="chip">no access</span> : <span className={'chip ' + (e === 1 ? 'ok' : e <= 0.25 ? 'fail' : 'global')}>{pct(e)} used</span>;

  return (
    <Bench
      title="Split the work, watch a warp"
      task="Find the one setting where the stores coalesce. Then watch what the searches cost."
      foot={
        <>
          The basic kernel (Fig. 13.9) on {fmtInt(total)} 4-byte ints, from src/sim/mergeaccess.ts, which counts segments with
          chapter 6's model ({SEGMENT_MODELS[model].facts}). One warp, one step of merge_sequential, no caches. The launch
          has ⌈total ÷ outputs per thread⌉ threads; line 03 then recomputes the outputs per thread from the thread count, as
          the kernel does.
        </>
      }
    >
      <div
        data-explorer="warp"
        data-threads={threads}
        data-per-thread={E}
        data-step={step}
        data-warp={w}
        data-writes={ws.writes.transactions}
        data-writes-eff={ws.writes.efficiency}
        data-reads-a={ws.readsA.transactions}
        data-reads-b={ws.readsB.transactions}
        data-paths={ws.paths.join(',')}
        data-search-reads={cost.searchReads}
        data-search-per-output={cost.searchPerOutput}
        data-unwritten={layout.unwritten.length}
      >
        <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
          <label className="field" style={{ maxWidth: '100%' }}>
            Data
            <select value={data} onChange={(e) => setData(e.target.value as DataId)} style={{ maxWidth: '100%' }}>
              {DATA.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
            </select>
          </label>
          <label className="field">
            m + n = {fmtInt(total)}
            <input type="range" min={64} max={1024} value={total} aria-label="total" onChange={(e) => setTotal(Number(e.target.value))} />
          </label>
          <label className="field">
            outputs per thread = {per}
            <input type="range" min={1} max={16} value={per} aria-label="outputs per thread" onChange={(e) => { setPer(Number(e.target.value)); setS(0); setWarp(0); }} />
          </label>
          <span className="field">
            Line 03
            <span className="seg" role="group" aria-label="Line 03">
              <button aria-pressed={formula === 'printed'} onClick={() => setFormula('printed')}>as printed</button>
              <button aria-pressed={formula === 'corrected'} onClick={() => setFormula('corrected')}>corrected</button>
            </span>
          </span>
        </div>
        <div className="row" style={{ marginBottom: '0.5rem', alignItems: 'flex-end' }}>
          <label className="field" style={{ maxWidth: '100%' }}>
            Warp width
            <select value={ww} onChange={(e) => { setWw(Number(e.target.value)); setWarp(0); }} style={{ maxWidth: '100%' }}>
              <option value={32}>32 lanes (NVIDIA; AMD RDNA)</option>
              <option value={64}>64 lanes (AMD CDNA)</option>
              {widths.filter((x) => x.width !== 32 && x.width !== 64).map((x) => (
                <option key={x.backend} value={x.width}>{x.width} lanes (your device: {x.device})</option>
              ))}
            </select>
          </label>
          <label className="field" style={{ maxWidth: '100%' }}>
            Segment model
            <select value={model} onChange={(e) => setModel(e.target.value as SegmentModelId)} style={{ maxWidth: '100%' }}>
              {Object.values(SEGMENT_MODELS).map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
            </select>
          </label>
          {SEGMENT_MODELS[model].segmentBytes == null ? (
            <label className="field">
              Your choice (not a fact)
              <select value={custom} onChange={(e) => setCustom(Number(e.target.value))}>
                {[32, 64, 128].map((v) => <option key={v} value={v}>{v} bytes</option>)}
              </select>
            </label>
          ) : null}
        </div>
        <p className="muted" style={{ margin: '0 0 0.7rem', fontSize: '0.84rem' }}>
          {widths.length
            ? <>Your device{widths.length > 1 ? 's' : ''} reported: {widths.map((x) => `${x.device} (${x.backend}), ${x.width} lanes`).join('; ')}. On Metal the width is read from the pipeline.</>
            : <>No warp width reported by your devices yet (lab 6.1 records it). On Metal the width is read from the pipeline, so here it is a choice.</>}
        </p>
        <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
          <label className="field">
            warp {w} of {warps} (threads {w * ww} to {Math.min(threads, (w + 1) * ww) - 1})
            <input type="range" min={0} max={warps - 1} value={w} aria-label="warp" disabled={warps === 1} onChange={(e) => setWarp(Number(e.target.value))} />
          </label>
          <label className="field">
            step s = {step} of merge_sequential (0 to {Math.max(0, E - 1)})
            <input type="range" min={0} max={Math.max(0, E - 1)} value={step} aria-label="step" disabled={E <= 1} onChange={(e) => setS(Number(e.target.value))} />
          </label>
        </div>

        <p className="narrate" style={{ margin: '0 0 0.8rem' }}>
          {fmtInt(threads)} threads. Line 03 {formula === 'printed' ? 'as printed' : 'corrected'} gives {E} output{E === 1 ? '' : 's'} per thread
          {formula === 'printed' ? ` (⌊${fmtInt(total)} ÷ ${fmtInt(threads)}⌋)` : ` (⌈${fmtInt(total)} ÷ ${fmtInt(threads)}⌉)`}
          {E !== per ? `, not the ${per} the launch was sized for` : ''}.{' '}
          {layout.unwritten.length ? (
            <b className="t-fail" data-unwritten-text>
              C[{fmtInt(layout.unwritten[0])}] to C[{fmtInt(total - 1)}], {fmtInt(layout.unwritten.length)} output{layout.unwritten.length === 1 ? '' : 's'}, are never written.
            </b>
          ) : (
            'Every output is written.'
          )}
        </p>

        <Strips layout={layout} warp={w} warpWidth={ww} s={step} />
        <p className="muted" style={{ margin: '0.3rem 0 0.8rem', fontSize: '0.8rem' }}>
          Bands: each thread's slice of A, B and C, in alternating shades; this warp's threads are tinted and outlined; the number is the array's length. Ticks: what this warp's lanes load and store at step {step}.
        </p>

        <div className="mx-label">Lanes at step {step}: A, B = main loop takes from A or B; a, b = copying the rest of A or B; · = nothing to do</div>
        <div className="mx" style={{ display: 'flex', flexWrap: 'wrap', maxWidth: '49rem', marginBottom: '0.8rem' }} data-lanes={ws.lanes.map((l) => PATH_LETTER[l.path]).join('')}>
          {ws.lanes.map((l) => (
            <span key={l.lane} style={{ minWidth: '1.4rem', height: '1.6rem' }}
              title={l.path === 'idle' ? `lane ${l.lane}: nothing to do` : `lane ${l.lane} (thread ${l.t}): ${PATH_WORDS[l.path as Exclude<Path, 'idle'>]}; loads ${l.reads.map((r) => `${r.arr}[${r.index}]`).join(' and ')}; stores C[${l.write}]`}>
              {PATH_LETTER[l.path]}
            </span>
          ))}
        </div>

        <div className="mx-label">One instruction of this warp at step {step}</div>
        {/* The efficiency chip sits beside the count so that it stays in view on a phone; the byte columns scroll. */}
        <div style={{ overflowX: 'auto', maxWidth: '44rem', margin: '0 0 0.6rem' }}>
          <table className="data" style={{ margin: 0, minWidth: '30rem', display: 'table' }} aria-label="This warp's accesses at this step: one instruction each">
            <thead>
              <tr><th>Access</th><th className="num">lanes</th><th className="num">{unit}s of {seg} B</th><th>used</th><th className="num">bytes fetched</th><th className="num">needed</th></tr>
            </thead>
            <tbody>
              {[
                { name: 'loads of A', a: ws.readsA, lanes: lanesA, key: 'a' },
                { name: 'loads of B', a: ws.readsB, lanes: lanesB, key: 'b' },
                { name: 'stores to C', a: ws.writes, lanes: lanesC, key: 'c' },
              ].map((r) => (
                <tr key={r.key} data-access={r.key}>
                  <td>{r.name}</td>
                  <td className="num">{r.lanes}</td>
                  <td className="num"><b>{r.a.transactions}</b></td>
                  <td data-eff>{effChip(r.a.efficiency, r.lanes)}</td>
                  <td className="num">{fmtInt(r.a.bytesFetched)}</td>
                  <td className="num">{fmtInt(r.a.bytesNeeded)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="row" style={{ margin: '0 0 0.6rem' }}>
          {/* Lanes in different loops diverge for certain; A versus B inside the main loop may compile to a select. */}
          <span className={'chip ' + (loops > 1 ? 'fail' : ws.paths.length > 1 ? '' : 'ok')} data-divergence={ws.paths.length} data-loops={loops}>
            {ws.paths.length} path{ws.paths.length === 1 ? '' : 's'} at this step
          </span>
          <span className="muted">{ws.paths.map((p) => PATH_WORDS[p]).join('; ')}</span>
        </p>
        <p className="row" style={{ margin: 0 }} data-cost>
          <span>co_rank: {fmtInt(cost.calls)} calls, <b>{fmtInt(cost.searchReads)}</b> global reads</span>
          <span>merge_sequential: {fmtInt(cost.mergeReads)} loads</span>
          <span className="chip global">{cost.searchPerOutput.toFixed(2)} search reads per output</span>
        </p>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions

const own: Record<string, Question> = {
  'ch13/13.3/q-wrong-split': {
    kind: 'choice',
    id: 'ch13/13.3/q-wrong-split',
    prompt: (
      <>
        Two threads split the running example at <code>k = 4</code> (thread 0 writes C[0..4)), and someone sets{' '}
        <code>i = 4, j = 0</code>. What goes wrong, and which test catches it?
      </>
    ),
    options: [
      { label: '7 lands after 9. The first test fires, A[3] = 9 > B[0] = 7: i is too high.', correct: true },
      { label: 'Nothing goes wrong: any split with i + j = k gives a sorted C.', why: 'i + j = k only fixes how many elements thread 0 takes, not which ones.' },
      { label: 'C stays sorted, but the tie between the two 7s now goes to B.', why: "Here a key is out of order, not just a tie: 9 is written before B's 7, and A's 7 still comes first." },
      { label: '10 lands before 8. The second test fires, B[j−1] ≥ A[i]: j is too high.', why: 'That is the other wrong guess, i = 2 and j = 2. Here j = 0, so the second test\'s bounds check j > 0 is false and it reads nothing.' },
    ],
    explain: (
      <p>
        A cut is right when nothing thread 0 takes is bigger than what thread 1 takes from the other array. With i = 4,
        thread 0 takes A[0..4), whose last element is 9, while thread 1 starts B at B[0] = 7. The first test compares exactly
        those two, A[i − 1] and B[j], finds 9 &gt; 7, and says i is too high. Moving i down (and j up) is the right
        direction; the co-rank is i = 3. In the explorer, choose "Try any split" at k = 4.
      </p>
    ),
    source: 'ch. 13 §13.4, pp. 309, 311',
  },
  'ch13/13.3/q-iterations': {
    kind: 'numeric',
    id: 'ch13/13.3/q-iterations',
    prompt: (
      <>
        <code>m = n = 1,048,576</code> and <code>k = 1,048,576</code>. At most how many times can co_rank's loop move{' '}
        <code>i</code>?
      </>
    ),
    answer: 21,
    unit: 'moves',
    explain: (
      <p>
        i can end anywhere from max(0, k − n) = 0 to min(k, m) = 1,048,576: that is 1,048,577 candidates. Every move at
        least halves what is left, so at most ⌈log2(1,048,577)⌉ = 21 moves, plus one last pass that confirms. The bound is
        reached when every element of B is below every element of A: i walks 1,048,576, 524,288, … 2, 1, 0. The book's
        log2 N (p. 313) is the same order; src/sim/corank.test.ts runs this exact case.
      </p>
    ),
    source: 'ch. 13 §13.4, p. 313; the exact bound is ours',
  },
  'ch13/13.3/q-tie-test': {
    kind: 'choice',
    id: 'ch13/13.3/q-tie-test',
    prompt: (
      <>
        A colleague writes the tests as <code>A[i-1] &gt;= B[j]</code> and <code>B[j-1] &gt; A[i]</code>, keeps the{' '}
        <code>&lt;=</code> in merge_sequential, and the merge test on plain ints passes. What is wrong?
      </>
    ),
    options: [
      { label: 'Nothing: either tie rule gives a valid split, and the test passed.', why: 'Each is valid alone. The co-rank and merge_sequential must use the same one.' },
      { label: 'The output is no longer sorted wherever a tie falls on a cut between threads.', why: 'The swapped search still finds a valid cut, just the one that sends ties to B; keys stay in order.' },
      { label: 'With >= in the first test, the search can read A[m] or B[n], past the end.', why: 'The bounds tests are unchanged and still come first, so j < n and i < m are checked before any read.' },
      { label: 'At the cuts, ties now go to B: the keys are sorted, but the merge is unstable.', correct: true },
    ],
    explain: (
      <p>
        The swapped tests converge to a different, equally consistent split: the one where B wins ties. Inside each thread
        nothing changed, so each slice is a stable merge of its own pieces, ties to A. Only at the cuts do equal keys swap
        sides. A test that merges ints compares keys and cannot tell two 10s apart; std::merge is guaranteed stable with
        the first range first, so comparing key–value pairs against it catches the bug. Part 6 of the animation
        "Searching for the split" shows it on the running example.
      </p>
    ),
    source: 'ch. 13 §13.4, p. 311; notes/today/ch13-facts.md A6',
  },
  'ch13/13.3/q-ceil-printed': {
    kind: 'numeric',
    id: 'ch13/13.3/q-ceil-printed',
    prompt: (
      <>
        Fig. 13.9 exactly as printed, with <code>m + n = 1,000</code> and 3 blocks of 128 threads. How many outputs are never
        written?
      </>
    ),
    answer: 232,
    unit: 'outputs',
    explain: (
      <p>
        384 threads. 1000 / 384 in integer division is 2; ceil(2) is 2. So 384 × 2 = 768 outputs are covered, and C[768] to
        C[999], 232 of them, keep whatever was there before. The corrected line, (m + n + T − 1) / T, gives 3, and the clamp
        on k_curr and k_next keeps the last threads inside C. Set the warp explorer below to "as printed" to see the same gap.
      </p>
    ),
    source: 'ch. 13 §13.5, Fig. 13.9 line 03, p. 313; notes/today/ch13-facts.md A4',
  },
  'ch13/13.3/q-coalesce': {
    kind: 'numeric',
    id: 'ch13/13.3/q-coalesce',
    prompt: (
      <>
        Basic kernel, 4-byte ints, 8 outputs per thread. At step s, lane t of a warp stores{' '}
        <code>C[k_curr + s] = C[8t + s]</code>. With 32-byte segments, how many segments does one 32-lane store touch?
      </>
    ),
    answer: 32,
    unit: 'segments',
    explain: (
      <p>
        Neighbouring lanes store 8 elements, 32 bytes, apart, so every lane lands in its own 32-byte segment: 32 segments
        for 128 useful bytes, 12.5 % of what is moved. At one output per thread the same store is 32 consecutive ints, 4
        segments, all of it used. The explorer below counts both.
      </p>
    ),
    source: 'ch. 13 §13.5, p. 314; notes/today/ch06-facts.md A2',
  },
  'ch13/13.3/q-one-per-thread': {
    kind: 'choice',
    id: 'ch13/13.3/q-one-per-thread',
    prompt: <>At one output per thread the basic kernel's stores coalesce perfectly. Why not always use one?</>,
    options: [
      { label: 'A grid with one thread per output is too large for the GPU to launch.', why: 'A grid of a million threads is ordinary.' },
      { label: 'With slices of one element each, the merge would no longer be stable.', why: 'Stability depends on the tie rule, not on the slice length.' },
      { label: 'Each output would pay for two whole binary searches on global memory.', correct: true },
      { label: 'The threads would race, since their searches read the same elements.', why: 'They only read A and B; nobody writes them, and reads do not race.' },
    ],
    explain: (
      <p>
        At one output per thread, every output pays for two binary searches on global memory, each about log2 N passes of
        up to four scattered reads. The book's §13.8 makes the point: the overhead of parallelising merge is the
        per-thread search, so each thread should produce many outputs, and every kernel in the chapter does. The explorer
        shows the trade: search reads per output fall roughly as one over the outputs per thread, while the stores spread
        out. §13.6, next, gets both by staging the inputs in shared memory.
      </p>
    ),
    source: 'ch. 13 §13.8, p. 327',
  },
};

export const questionsS133: Record<string, Question> = { ...own, ...questionsStaging };

// ---------------------------------------------------------------- the page

export function S133() {
  return (
    <>
      <p>
        Every kernel so far could tell from its own index which inputs it needs. A merge cannot. A thread that has to
        produce outputs 300 to 307 cannot know whether they come from A, from B or from both until it has looked at the
        values. This chapter answers that with one function, the co-rank, and this page is built around it: what it
        computes, how it finds the answer in a few steps, how a kernel uses it, what it costs in memory traffic, and how
        staging the inputs in shared memory cuts that cost.
      </p>
      <Source pages="303, 308" />
      <p>
        The running example is the book's: A = [1, 7, 8, 9, 10] (m = 5) and B = [7, 10, 10, 12] (n = 4) merge into C =
        [1, 7, 7, 8, 9, 10, 10, 10, 12]. The merge is stable: equal keys keep their order, and when A and B hold the same key,
        A's element goes first, because <code>merge_sequential</code> takes A[i] when <code>A[i] &lt;= B[j]</code>. Keep that{' '}
        <code>&lt;=</code> in mind. It decides how the search has to be written.
      </p>
      <Source section="13.1, 13.2" pages="304–305" figs="Figs. 13.1, 13.2" />

      <h2>Split the output, not the input</h2>
      <p>
        Pick any position k in C, from 0 up to and including m + n. The first k outputs are the merge of a prefix of A and a
        prefix of B: the first i elements of A and the first j of B, with i + j = k. For every k there is exactly one such
        pair (the book credits the proof to Siebert and Träff). The book calls k the <b>rank</b> of C[k], and i and j its{' '}
        <b>co-ranks</b>.
      </p>
      <Source section="13.3" pages="306–308" figs="Fig. 13.3" />
      <p>
        Two cases from the book. C[4] = 9 has co-ranks (3, 1): the first four outputs are A[0..3) and B[0..1). C[6], B's
        first 10, has co-ranks (5, 1): the first six outputs take all five elements of A and one of B. The range of k
        includes m + n, one past the end of C. Its co-ranks are (m, n), and that query is how the last thread finds where
        its inputs stop.
      </p>
      <Aside>
        <p>
          Fig. 13.3(b) labels the C[6] case i = 4, j = 2. That pair also adds up to 6, but it puts B's 10 ahead of A's 10,
          against the tie rule. The text (pp. 307–308) and the code give (5, 1), and so does the explorer below.
        </p>
      </Aside>
      <p>
        So split the output. Cut C into equal slices, one per thread. Thread t knows that its slice starts at k = t ×
        (outputs per thread) without looking at any data. It calls the co-rank function at its own start and at the next
        thread's start, and the two pairs it gets back bracket its inputs. From then on threads touch disjoint pieces of A, B
        and C, and each runs the ordinary sequential merge on its own pieces. No thread waits for another.
      </p>
      <Today checked="2026-11-20 (notes/today/ch13-facts.md, A1)">
        <p>
          The code you will read uses other names. CUB and rocPRIM call this function merge path (CUB cites Odeh et al.,
          2012); MLX and PyTorch's MPS backend call it <code>merge_partition</code>; llama.cpp's Metal merge kernel comments it as a
          "binary-search partition (i, j) such that i + j = k".
        </p>
      </Today>
      <p>
        <a href={href('ch22', '22.2')}>Chapter 22, §22.2 "Problem decomposition"</a>, gives this choice a name: an
        output-centric decomposition. Atomics are not the reason for it here, since
        every output comes from exactly one input either way. Load balance is: every thread gets the same number of
        outputs whatever the data, and only the input slices vary. Cut the inputs instead, and threads could end up merging
        very different numbers of elements.
      </p>
      <Source section="22.2" pages="535" />

      <h2>Find the split by binary search</h2>
      <p>
        Why a search and not a formula? Take two threads splitting the running example at k = 4. Try i = 4, j = 0: thread 0
        gets A[0..4), which ends with 9, while B[0] = 7 goes to thread 1, so 7 lands after 9. Try i = 2, j = 2: thread 0 gets
        B[1] = 10 while A[2] = 8 goes to thread 1, so 10 lands before 8. Only i = 3 works. Each wrong guess is wrong in a
        direction you can read off the elements on either side of the cut, and that is what makes it a binary search.
      </p>
      <Source section="13.4" pages="308–309" figs="Fig. 13.4" />
      <p>
        The two tests say it precisely. A candidate (i, j) is right when nothing thread 0 takes is bigger than what thread 1
        takes from the other array: <code>A[i−1] &lt;= B[j]</code> and <code>B[j−1] &lt; A[i]</code>. If A[i−1] &gt; B[j], i
        is too high; if B[j−1] &gt;= A[i], j is too high. Why one is strict and the other not is the subject of the next
        part. Each test first checks that the elements it compares exist (i &gt; 0 and j &lt; n, or j &gt; 0 and i &lt; m),
        and relies on <code>&amp;&amp;</code> not evaluating its right side when the left is false.
      </p>
      <Source section="13.4" pages="311" />
      <p>
        Where the search starts. i cannot exceed m or k, so it starts at its largest possible value, min(k, m), with j = k −
        i. At most n of the first k outputs can come from B, so i is at least k − n: that is i_low = max(0, k − n), and
        likewise j_low = max(0, k − m). Every move changes i and j by the same delta in opposite directions, so i + j = k
        holds throughout. delta is the ceiling of half the distance to the lower bound, <code>(i - i_low + 1) &gt;&gt; 1</code>,
        so it is never 0 when a test fires.
      </p>
      <Source section="13.4" pages="309–311" figs="Fig. 13.5" />
      <Code
        tabs={[{ label: 'co_rank', code: CO_RANK, tiers: MERGE_TIERS, status: SNIPPET_STATUS }]}
        caption={
          <>
            Fig. 13.5 with <code>__device__</code>, <code>const int*</code>, and its two comment slips fixed (the j_low line's
            comment said i_low; the delta comment had unbalanced parentheses). Line numbers are the book's. From{' '}
            <code>labs/ch13/snippets/merge_basic.cu</code>.
          </>
        }
      />
      <p>
        How many moves? Each move at least halves the distance i can still travel, so for long inputs the book counts
        log2 N steps, N being the length of the longer input. Each pass round the loop reads at most four elements of
        global memory.
      </p>
      <Source section="13.4" pages="313" />
      <p>
        A tighter count: i starts in a window of w + 1 candidates, with w = min(k, m) − max(0, k − n), and the loop moves i
        at most ⌈log2(w + 1)⌉ times, then makes one last pass that confirms. The bound is reached, for instance, when every
        element of B is below every element of A.
      </p>
      <Beyond>
        <p>
          The exact bound is ours. <code>src/sim/corank.test.ts</code> checks it on every pair of sorted arrays up to length
          5 with keys 0 to 2 and on random arrays up to length 40, and finds inputs that reach it for every window size.
        </p>
      </Beyond>
      <p>
        The explorer runs Fig. 13.5 on arrays you choose. Start with the running example at k = 3, the search the book
        traces in Figs. 13.6 to 13.8. Turn on predict mode and call each step before you see it. Then try all keys equal,
        the two one-sided presets and A empty, and drag k to both ends. Finally choose "Try any split" and force a wrong i,
        as in Fig. 13.4.
      </p>
      <Simplification>
        <p>
          The explorer runs Fig. 13.5's integer logic exactly, on small arrays: the same tests, the same short-circuit
          reads, the same moves. It has no memory, no threads and no timing.
        </p>
      </Simplification>
      <CoRankExplorer />
      <Ask q={own['ch13/13.3/q-wrong-split']} />
      <Ask q={own['ch13/13.3/q-iterations']} label="Work it out" />
      <p>
        The animation shows the same search once, for thread 1 of three, then every thread's search side by side, then what
        changes if you swap the two tests.
      </p>
      <Simplification>
        <p>
          Every bar, test and slice is computed by the explorer's model (src/sim/corank.ts), not drawn by hand. The threads
          appear to search in step; real threads run at their own pace, which does not matter here because no thread reads
          anything another writes.
        </p>
      </Simplification>
      <CoRankScene />

      <h2>Ties decide which split is right</h2>
      <p>
        Look at the tests again: <code>&gt;</code> in the first, <code>&gt;=</code> in the second. They are lopsided on
        purpose. Inside one thread, merge_sequential sends ties to A. The co-rank must make the same choice at the cuts
        between threads: on a tie, A's element belongs to the earlier slice. So the first test does not call i too high
        when A[i−1] equals B[j], and the second does call j too high when B[j−1] equals A[i].
      </p>
      <Source section="13.1, 13.2, 13.4" pages="304, 305, 311" />
      <p>
        Swap the operators and the search still converges, to the split that sends ties to B. Each thread still breaks ties
        toward A inside its slice, so the keys come out sorted and a test that merges plain ints passes. But wherever a tie
        falls on a cut, equal keys from the two arrays trade places. If the lists were sorted earlier by another key, the
        order that sort produced is gone, and that order is what stability exists to keep. Only a test with tagged values
        can see the difference, which is why lab 13.1 merges key–value pairs. Part 6 of the animation above shows it at
        the running example's cuts, and the explorer's "swap the tie tests" box shows the search that gets there.
      </p>
      <Today checked="2026-11-20 (notes/today/ch13-facts.md, A6, A7, B2, C4)">
        <p>
          <code>std::merge</code> is guaranteed stable, first range first. <code>thrust::merge</code> and rocThrust are
          documented stable. <code>cub::DeviceMerge</code> (CCCL 2.7.0, CUDA 12.8) is documented "Merging is unstable",
          although its code sends ties to the first sequence: that last point is our reading. hipCUB's DeviceMerge calls
          rocprim::merge, which says nothing about stability. Apple documents no GPU merge.
        </p>
      </Today>
      <Ask q={own['ch13/13.3/q-tie-test']} />

      <h2>The whole kernel in eleven lines</h2>
      <p>
        With co_rank in hand the kernel is short. Each thread works out how many outputs every thread owns, where its own
        first output and the next thread's first output are, calls co_rank on both, and hands its two input slices to
        merge_sequential. Neighbouring threads compute the same boundary twice (thread t's k_next is thread t + 1's k_curr);
        that duplication is what lets them run without talking to each other.
      </p>
      <Source section="13.5" pages="313" figs="Fig. 13.9" />
      <Code
        tabs={[
          { label: 'merge_basic_kernel', code: MERGE_BASIC, tiers: MERGE_TIERS, status: SNIPPET_STATUS },
          { label: 'merge_sequential', code: MERGE_SEQ, tiers: MERGE_TIERS, status: SNIPPET_STATUS },
          { label: 'co_rank', code: CO_RANK, tiers: MERGE_TIERS, status: SNIPPET_STATUS },
        ]}
        caption={
          <>
            Figs. 13.9, 13.2 and 13.5, corrected, with the book's line numbers. Line 03 uses integer ceiling division; line
            04 clamps k_curr as line 05 clamps k_next (ours); min is a ternary (see "On HIP and Metal"); both helpers are{' '}
            <code>__device__</code>. From <code>labs/ch13/snippets/merge_basic.cu</code>.
          </>
        }
      />
      <p>
        Line 03 as printed is <code>{LINE03_PRINTED}</code>. Both operands of the division are integers, so the quotient is
        truncated before <code>ceil</code> sees it, and <code>ceil</code> has nothing left to round. Whenever m + n is not a
        multiple of the thread count, each thread takes one output too few and the last outputs of C are never written.
        Write <code>(a + b - 1) / b</code>, or <code>cuda::ceil_div</code> on CUDA.
      </p>
      <Today checked="2026-11-20 (notes/today/ch13-facts.md, A4, C2; ch02-facts A8)">
        <p>
          C++ fixes this: for integers, <code>/</code> "yields the algebraic quotient with any fractional part discarded".
          nvcc and HIP's Clang follow the standard, and MSL divides integers the same way.
        </p>
      </Today>
      <Ask q={own['ch13/13.3/q-ceil-printed']} label="Work it out" />
      <p>
        The book's worked example runs three threads on the running example, three outputs each. Thread 1 has k_curr = 3
        and k_next = 6; co_rank gives i_curr = 2 and i_next = 5, so j_curr = j_next = 1, and it calls merge_sequential with
        (&amp;A[2], 3, &amp;B[1], 0, &amp;C[3]). Its B slice is empty: all three of its outputs come from A, and
        merge_sequential's first loop never runs. The table is computed by the same model as the explorers.
      </p>
      <Source section="13.5" pages="314" />
      <WorkedExample />

      <h2>What it costs</h2>
      <p>
        The kernel is correct and short, and it uses memory badly in two ways. Inside merge_sequential, neighbouring threads
        work on slices that are a whole slice apart, so at each step the lanes of a warp store to addresses that far apart,
        and load from A and B at places that depend on the data. And every thread runs two binary searches on global memory,
        whose reads jump around and seldom share a segment with a neighbour's.
      </p>
      <Source section="13.5" pages="314" />
      <p>
        Coarsening pulls both ways. More outputs per thread means fewer threads, so fewer searches per output: one search
        is paid for by all of a thread's outputs. But it also spreads each warp's stores further apart. At one output per
        thread the stores coalesce perfectly and every output pays for two searches; at eight, the searches are cheap per
        output and every lane stores into its own segment. The tiled kernel below is the book's way to get both.
      </p>
      <Source section="13.8" pages="327" />
      <p>
        A third cost may hide in the branches. At each step of merge_sequential every lane decides between A and B on its
        own data, and some lanes are already in one of the tail loops. Lanes in different loops take different paths for
        certain, and <a href={href('ch04', '4.5')}>chapter 4</a> showed what that costs: the warp runs each path in turn.
        The main loop's choice between A and B is only two assignments, which a compiler may turn into a select, with no
        divergence at all.
      </p>
      <Source section="4.5" pages="79–82" />
      <Beyond>
        <p>
          Ours: the book lists two memory costs for this kernel (p. 314) and does not discuss divergence. Whether the A
          or B choice compiles to a select is not checked here.
        </p>
      </Beyond>
      <p>
        The explorer below runs the basic kernel on up to 1,024 outputs and shows one warp at one step. Answer this
        before you touch it.
      </p>
      <Ask q={own['ch13/13.3/q-coalesce']} label="Before you touch it" />
      <p>
        Then find the setting where the stores coalesce, watch the search reads per output as you change the outputs per
        thread, and switch line 03 to "as printed" with a total that is not a multiple of the thread count.
      </p>
      <Simplification>
        <p>
          One instruction at a time, no caches, and the warp idealised as running step s together on every lane, whatever
          path each lane is on. Each thread walks its own slice in order, so a real L1 or L2 may serve later steps from
          lines fetched earlier; only lab 13.1 measures that. Apple documents no segment size, so for Metal it is your choice.
        </p>
      </Simplification>
      <WarpExplorer />
      <Ask q={own['ch13/13.3/q-one-per-thread']} />

      <StagingPart />

      <h2>Where you will meet it</h2>
      <p>
        Sampling. A top-k over a vocabulary-sized row can be done block by block: each block keeps its own top k, sorted,
        and then pairs of those lists are merged, producing only the first k outputs of each merge. That combining step is
        this chapter's merge, and a thread whose slice starts at or past k has nothing to do and exits.
      </p>
      <Beyond>
        <p>
          This bridge is ours; the book does not discuss sampling. Checked against the projects' source on 2026-11-20
          (notes/today/ch13-facts.md, D1 to D4).
        </p>
      </Beyond>
      <p>
        llama.cpp's Metal backend does exactly this for short rows or small k. A bitonic sort per threadgroup keeps each
        block's top k, then <code>kernel_argsort_merge_f32_i32</code> merges pairs of runs straight from device memory, one
        chunk per thread, with a binary-search partition: the shape of Fig. 13.9, ties to the first run, and threads past
        top_k returning at once. Past a size threshold (long rows with a large k, or many long rows) it switches to radix
        select. MLX's <code>topk</code> and{' '}
        <code>argpartition</code> on Metal run a full merge sort, so mlx-lm's top-k and top-p sampling go through this
        search too. On the CUDA side, llama.cpp, PyTorch and CUB do top-k with radix select, bitonic sort or{' '}
        <code>cub::DeviceTopK</code>, not with a merge.
      </p>

      <h2>On HIP and Metal</h2>
      <p>
        One <code>.cu</code> file serving CUDA and HIP through <code>pmpp/vendors/hip.h</code> works, by the
        documentation, for threads, blocks, shared memory, barriers, legacy atomics and runtime calls. It does not
        extend to warp <code>_sync</code> intrinsics (HIP wants a 64-bit lane mask), cooperative groups and
        half-precision headers (different include paths), CUB functors, tensor-core intrinsics or <code>__dp4a</code>.
        This chapter's kernels use only threads, shared memory and barriers, so they stay on the side that carries over,
        with one caveat of their own: integer <code>min</code>. That path has not been compiled for AMD hardware here
        yet. Metal changes more, and mostly about pointers.
      </p>
      <Beyond>
        <p>
          The book uses CUDA only. What carries over to HIP is notes/today/infra-facts.md's "Where the backends really
          differ"; everything in this table comes from notes/today/ch13-facts.md's table of the same name and the files it
          cites. None of it has been run on AMD hardware.
        </p>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th>Concern</th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>Qualifier on co_rank and merge_sequential</td>
            <td><code>__device__</code> (or <code>__host__ __device__</code>)</td>
            <td>same</td>
            <td>none: every function in a <code>.metal</code> file is GPU code</td>
          </tr>
          <tr>
            <td>One helper for global and on-chip data</td>
            <td>yes: one <code>const int*</code> helper</td>
            <td>yes</td>
            <td>no: address spaces are disjoint, so a <code>device</code> and a <code>threadgroup</code> overload, or a template on the pointer type (MLX writes <code>merge_partition</code> twice; PyTorch templates it)</td>
          </tr>
          <tr>
            <td>Integer <code>min</code> / <code>max</code></td>
            <td><code>__device__ int min(int, int)</code>, plus mixed signed/unsigned overloads that return <code>unsigned</code></td>
            <td>from Clang's header: same-type overloads and a template; no mixed device overloads</td>
            <td><code>T min(T, T)</code>: both arguments one type</td>
          </tr>
          <tr>
            <td>Bounds tests before the reads</td>
            <td><code>&amp;&amp;</code> short-circuits (a C++ guarantee)</td>
            <td>same</td>
            <td>not restated in the MSL spec; llama.cpp and MLX guard each read explicitly</td>
          </tr>
          <tr>
            <td>Two input tiles on chip</td>
            <td>one <code>extern __shared__</code> array split in two; bytes as the third launch argument</td>
            <td>same</td>
            <td><code>[[threadgroup(n)]]</code> buffers sized with <code>setThreadgroupMemoryLength</code>, in multiples of 16 bytes</td>
          </tr>
          <tr>
            <td>Barrier</td>
            <td><code>__syncthreads()</code></td>
            <td>same</td>
            <td><code>threadgroup_barrier(mem_flags::mem_threadgroup)</code></td>
          </tr>
          <tr>
            <td>Integer division (line 03)</td>
            <td>truncates</td>
            <td>same</td>
            <td>truncates; an out-of-range quotient is unspecified</td>
          </tr>
        </tbody>
      </table>
      <p>
        Two habits follow. Write min and max as ternaries, as Fig. 13.5 already does: a ternary means the same thing on
        CUDA, HIP, the course's emulator and MSL, while a mixed call such as <code>min(blockIdx.x * x, m + n)</code> may
        resolve to an <code>unsigned</code> overload on CUDA and to no single overload on HIP. And in MSL, guard each read
        with its own condition rather than leaning on the order of <code>&amp;&amp;</code>; it costs nothing and is what
        shipping Metal code does. Lab 13.1 has the Metal versions of these kernels beside the CUDA one: see the{' '}
        <a href={href('ch13', 'labs')}>labs page</a>.
      </p>
      <Today checked="2026-11-20 (notes/today/ch13-facts.md, A5, B1, C1 to C3)">
        <p>
          The HIP point is a reading of overload resolution, not a compiled result: the header has no mixed signed/unsigned
          device overload, so a mixed call needs a conversion on every candidate.
        </p>
      </Today>
    </>
  );
}
