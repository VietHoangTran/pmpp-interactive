import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, NoteGroup, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  EXERCISE_1,
  FIG_11_3,
  KIND_LABEL,
  SCHEDULES,
  SCHEDULE_LABEL,
  additionsAt,
  divergence,
  finalCount,
  iterations,
  kernelSource,
  levels,
  naiveAdditions,
  run,
  strides,
  totalAdditions,
  treeOf,
  warpStates,
  type Config,
  type Schedule,
} from '../../sim/ksblock';
import { functionSource } from '../../sim/toolchain';
import { KsDoublingScene } from './KsDoublingScene';
import { KsGrid, TraceView, fromLevels } from './KsGrid';

const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const WRAP: CSSProperties = { flexWrap: 'wrap' };

// ---------------------------------------------------------------- stride by stride
const SIXTEEN = [3, 1, 7, 0, 4, 1, 6, 3, 5, 2, 8, 1, 0, 6, 2, 4];
const ROWS = [
  { id: 'ex1', label: 'exercise 1', values: EXERCISE_1 },
  { id: 'sixteen', label: 'sixteen values', values: SIXTEEN },
];
function parseRow(s: string): number[] | null {
  const parts = s.split(/[\s,]+/).filter(Boolean);
  if (parts.length < 2 || parts.length > 16) return null;
  const v = parts.map((p) => Number(p.replace('−', '-')));
  return v.every((x) => Number.isInteger(x) && Math.abs(x) <= 999) ? v : null;
}

function StrideStepper() {
  const [values, setValues] = useState<number[]>(EXERCISE_1);
  const [text, setText] = useState(EXERCISE_1.join(' '));
  const lv = useMemo(() => levels(values), [values]);
  const B = values.length;
  const L = lv.length - 1;
  const [k, setK] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [tree, setTree] = useState<number>(B - 1);
  useEffect(() => {
    setK(0);
    setTree(values.length - 1);
    setPlaying(false);
  }, [values]);
  useEffect(() => {
    if (!playing) return;
    if (k >= L) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setK((x) => x + 1), 1100);
    return () => window.clearTimeout(id);
  }, [playing, k, L]);
  const t = useMemo(() => (tree >= 0 ? treeOf(tree, B) : undefined), [tree, B]);
  const g = fromLevels(lv, k, { tree: t, hot: k, carries: true });
  const done = lv.slice(1, k + 1).reduce((s, l) => s + l.added.filter(Boolean).length, 0);
  const bad = parseRow(text) === null;
  const pow2 = Number.isInteger(Math.log2(B));

  return (
    <Bench
      title="Kogge-Stone, stride by stride"
      task="Step through the strides. Before each step, say which positions will be final after it."
      foot={
        <>
          All positions of a step add at the same time, reading the row above (the lockstep picture of Fig. 11.2). Shaded
          cells hold their final value. Under each value: the inputs it counts.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ flex: '1 1 16rem' }}>
          Input (2 to 16 whole numbers)
          <input type="text" value={text} aria-label="Input" aria-invalid={bad}
            style={{ width: '100%', fontVariantNumeric: 'tabular-nums', borderColor: bad ? 'var(--fail)' : undefined }}
            onChange={(e) => {
              setText(e.target.value);
              const v = parseRow(e.target.value);
              if (v) setValues(v);
            }} />
        </label>
        <span className="seg" role="group" aria-label="Rows" style={WRAP}>
          {ROWS.map((r) => (
            <button key={r.id} onClick={() => { setValues(r.values); setText(r.values.join(' ')); }}>{r.label}</button>
          ))}
        </span>
        <label className="field">
          Show the tree behind
          <select value={tree} onChange={(e) => setTree(Number(e.target.value))} aria-label="Tree behind output">
            <option value={-1}>no output</option>
            {values.map((_, i) => <option key={i} value={i}>y{i}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setK(0); }} disabled={k === 0}>Restart</button>
        <button onClick={() => { setPlaying(false); setK(k - 1); }} disabled={k === 0}>Back</button>
        <button className="primary" onClick={() => { setPlaying(false); setK(k + 1); }} disabled={k >= L}>Step</button>
        <button onClick={() => setPlaying(!playing)} disabled={k >= L}>{playing ? 'Pause' : 'Play'}</button>
        <span className="muted" data-readout="level">{k === 0 ? 'the input' : `after stride ${lv[k].stride} (step ${k - 1})`}</span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <KsGrid rows={g.rows} labels={g.labels} arrows={g.arrows} marks={g.marks} cellW={B > 8 ? 50 : 66} label="Kogge-Stone, level by level" />
      </div>
      <p style={{ marginBottom: 0 }} data-readout="stride" data-final={finalCount(B, k)} data-additions={done}>
        {k === 0 ? (
          <>Position 0 already holds its answer, y0 = x0. Every other position holds only its own input.</>
        ) : (
          <>
            Stride {lv[k].stride}: {additionsAt(B, lv[k].stride!)} additions, {done} so far. The first{' '}
            <b>{finalCount(B, k)}</b> positions are final; every other one holds the {2 ** k} inputs ending at it.
          </>
        )}{' '}
        {k === L ? (
          <>
            Done in {L} steps with <b>{totalAdditions(B)}</b> additions
            {pow2 ? ` = ${B}·${L} − ${B - 1}` : ''}, against {B - 1} for the loop.
          </>
        ) : null}
        {t && tree > 0 && k === L ? <> The tree behind y{tree} uses {t.nodes.filter((nd) => nd.level > 0 && nd.pos >= 2 ** (nd.level - 1)).length} of them.</> : null}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- which warps diverge
const SIZES = [64, 128, 256, 512, 1024];
function WarpExplorer() {
  const [B, setB] = useState(256);
  const [width, setWidth] = useState(32);
  const ss = strides(B);
  const [si, setSi] = useState(2);
  const s = ss[Math.min(si, ss.length - 1)];
  const ws = warpStates(B, width, s);
  const d = divergence(B, width);
  const W = 640;
  const bw = Math.min(80, (W - 8) / ws.length);
  return (
    <Bench
      title="Which warps diverge?"
      task="Find every stride at which some warp runs both paths of the if. Then change the warp width."
      foot={<>A warp diverges when some of its lanes pass <code>threadIdx.x &gt;= stride</code> and some do not. Lanes are numbered in thread order.</>}
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          Threads per block
          <select value={B} onChange={(e) => setB(Number(e.target.value))} aria-label="Threads per block">
            {SIZES.map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>
        <span style={FIELD}>
          Warp width
          <span className="seg" role="group" aria-label="Warp width">
            <button aria-pressed={width === 32} onClick={() => setWidth(32)}>32: the book, NVIDIA</button>
            <button aria-pressed={width === 64} onClick={() => setWidth(64)}>64: AMD gfx9</button>
          </span>
        </span>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          stride = {s}
          <input type="range" min={0} max={ss.length - 1} value={Math.min(si, ss.length - 1)} aria-label="Stride"
            onChange={(e) => setSi(Number(e.target.value))} style={{ width: '100%' }} />
        </label>
      </div>
      <svg viewBox={`0 0 ${W} 64`} role="img" aria-label="Warps of the block" data-part="warps" style={{ width: '100%', maxWidth: W, display: 'block' }}>
        {ws.map((w) => (
          <g key={w.warp} data-part="warp" data-warp={w.warp} data-state={w.state} data-active={w.activeLanes}>
            <rect x={4 + w.warp * bw} y={8} width={bw - 3} height={30} rx={2} fill="var(--cell)"
              stroke={w.state === 'divergent' ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={w.state === 'divergent' ? 2.2 : 1}
              strokeDasharray={w.state === 'idle' ? '3 2' : undefined} />
            <rect x={4 + w.warp * bw + ((bw - 3) * (w.width - w.activeLanes)) / w.width} y={8} width={((bw - 3) * w.activeLanes) / w.width} height={30} rx={2} fill="var(--ink-3)" opacity={0.55} />
            {ws.length <= 16 || w.warp % 4 === 0 ? (
              <text x={4 + w.warp * bw + (bw - 3) / 2} y={54} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{w.warp}</text>
            ) : null}
          </g>
        ))}
      </svg>
      <p className="muted" style={{ margin: '0.2rem 0 0.6rem' }}>
        Grey fill: the lanes of a warp that pass the test (the low lanes drop out first). Heavy outline: divergent.
        Dashed: the whole warp skips the addition.
      </p>
      <p style={{ margin: 0 }} data-readout="warps" data-divergent={ws.filter((w) => w.state === 'divergent').length}>
        At stride {s}:{' '}
        {ws.some((w) => w.state === 'divergent')
          ? `warp 0 has ${ws[0].activeLanes} of ${ws[0].width} lanes active, so it runs both paths; every other warp is full.`
          : `no warp is divergent: ${ws.filter((w) => w.state === 'idle').length} whole warps skip the addition and the rest are full.`}
      </p>
      <p style={{ marginBottom: 0 }} data-readout="loop" data-iterations={d.iterations} data-pairs={d.pairs}>
        Over the whole loop ({ss.length} iterations), some warp diverges in <b>{d.iterations}</b> iterations: strides{' '}
        {d.strides.join(', ')}, always warp 0 ({d.pairs} divergent warp-iterations in all).
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the race behind barrier 2
function RaceHunt() {
  const [b1, setB1] = useState(true);
  const [b2, setB2] = useState(false);
  const [direct, setDirect] = useState(false);
  const [schedule, setSchedule] = useState<Schedule>('book-race');
  const cfg: Config = useMemo(() => ({ ...FIG_11_3, threads: 8, barrier1: b1, barrier2: b2, temp: !direct }), [b1, b2, direct]);
  const trace = useMemo(() => run(cfg, EXERCISE_1, schedule), [cfg, schedule]);
  return (
    <Bench
      title="The race behind barrier 2"
      task="Barrier 2 starts removed. Press Go to the first failure, then try the other thread orders."
      foot={
        <>
          Fig. 11.3 on one block of 8 threads, exercise 1's values. The kernel shown is the text of the variant you
          picked; the four named variants are in labs/ch11/snippets and run on the course's emulator with the same
          results.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="check"><input type="checkbox" checked={b1} onChange={(e) => setB1(e.target.checked)} /> barrier 1 (line 10)</label>
        <label className="check"><input type="checkbox" checked={b2} onChange={(e) => setB2(e.target.checked)} /> barrier 2 {direct ? '(after the store)' : '(line 14)'}</label>
        <label className="check"><input type="checkbox" checked={direct} onChange={(e) => setDirect(e.target.checked)} /> add straight into buffer_s, no temp</label>
        <label className="field" style={{ maxWidth: '100%' }}>
          Thread order
          <select value={schedule} onChange={(e) => setSchedule(e.target.value as Schedule)} aria-label="Thread order">
            {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
          </select>
        </label>
      </div>
      <TraceView trace={trace} />
    </Bench>
  );
}

const VARIANTS: { label: string; over: Partial<Config> }[] = [
  { label: 'Fig. 11.3 as printed', over: {} },
  { label: 'barrier 2 removed', over: { barrier2: false } },
  { label: 'barrier 1 removed', over: { barrier1: false } },
  { label: 'straight into buffer_s, both barriers', over: { temp: false } },
];
function Outcomes() {
  return (
    <Bench
      title="Every thread order at once"
      task="Before reading the table: which order will hide every bug?"
      foot="Each cell runs the whole block in the model above and compares the output with the scan of exercise 1."
    >
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-part="outcomes">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>Kernel</th>
              {SCHEDULES.map((s) => <th key={s}>{SCHEDULE_LABEL[s]}</th>)}
            </tr>
          </thead>
          <tbody>
            {VARIANTS.map((v) => (
              <tr key={v.label}>
                <td style={{ paddingLeft: '0.6rem' }}>{v.label}</td>
                {SCHEDULES.map((s) => {
                  const r = run({ ...FIG_11_3, threads: 8, ...v.over }, EXERCISE_1, s);
                  const kinds = [...new Set(r.violations.map((x) => x.kind))];
                  return (
                    <td key={s} data-outcome={r.correct ? 'correct' : 'wrong'} data-wrong={r.wrong.length}>
                      {r.correct ? <span className="chip ok">correct</span> : <span className="chip fail">{r.wrong.length} of 8 wrong</span>}
                      {kinds.map((k) => <div key={k} className="muted">{KIND_LABEL[k]}</div>)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS112: Record<string, Question> = {
  'ch11/11.2/q-naive-work': {
    kind: 'numeric',
    id: 'ch11/11.2/q-naive-work',
    prompt: (
      <>
        The naive parallel scan gives each of 1024 threads its own output, and each thread adds up its prefix with a
        sequential loop. How many additions do the threads do in total?
      </>
    ),
    answer: naiveAdditions(1024),
    unit: 'additions',
    explain: (
      <p>
        Thread <i>i</i> does <i>i</i> additions, so the total is 0 + 1 + … + 1023 = 1024·1023/2 = 523,776, against 1023
        for the loop. And it is not even faster: the last thread alone does 1023 additions in a row, so the span is the
        loop's.
      </p>
    ),
    source: 'ch. 11 §11.2, p. 254',
  },
  'ch11/11.2/q-ex1': {
    kind: 'numeric',
    id: 'ch11/11.2/q-ex1',
    prompt: (
      <>
        Exercise 1: Kogge-Stone on 4 6 7 1 2 8 5 2. Which value is at position 5 after the stride-2 step? (Positions count
        from 0.)
      </>
    ),
    answer: levels(EXERCISE_1)[2].values[5],
    explain: (
      <p>
        After stride 1: 4 10 13 8 3 10 13 7. After stride 2 each position adds the value two to its left: position 5 =
        10 + 8 = 18, which counts x2 to x5 (7 + 1 + 2 + 8). After stride 4 the array is 4 10 17 18 20 28 33 35.
      </p>
    ),
    source: 'ch. 11 exercise 1, p. 286',
  },
  'ch11/11.2/q-final': {
    kind: 'numeric',
    id: 'ch11/11.2/q-final',
    prompt: <>A 16-element Kogge-Stone scan has run the strides 1, 2 and 4. How many positions hold their final value?</>,
    answer: finalCount(16, 3),
    unit: 'positions',
    explain: (
      <p>
        After <i>k</i> steps position <i>i</i> holds up to 2<sup>k</sup> inputs ending at <i>i</i>. Positions 0 to 7 have
        at most 8 inputs to count, so after 3 steps they are complete; positions 8 to 15 still wait for the stride-8 step.
      </p>
    ),
    source: 'ch. 11 §11.2, pp. 255–256',
  },
  'ch11/11.2/q-race': {
    kind: 'choice',
    id: 'ch11/11.2/q-race',
    prompt: (
      <>
        Barrier 2 is removed and, in the stride-2 iteration, thread 4 stores its new value before thread 6 reads{' '}
        <code>buffer_s[4]</code>. Which inputs does position 6 count twice at the end?
      </>
    ),
    options: [
      { label: 'x1 and x2', correct: true },
      { label: 'x3 and x4', why: 'Those are what thread 6 should have read. It read more than that.' },
      { label: 'x0, x1 and x2', why: 'x0 arrives once, in the stride-4 step.' },
      { label: 'none: thread 6 just loses x3 and x4', why: 'It still gets x3 and x4, inside the larger value thread 4 stored.' },
    ],
    explain: (
      <p>
        Thread 6 should add x3..x4 to its x5..x6. It finds thread 4's new x1..x4 instead and stores x1..x6. The stride-4
        step then adds position 2, which holds x0..x2, so x1 and x2 arrive a second time: x0 + 2x1 + 2x2 + x3 + x4 + x5 +
        x6. The book prints the result without x0, which looks like a slip; the point, x1 and x2 counted twice, stands.
      </p>
    ),
    source: 'ch. 11 §11.2, pp. 257–258; the missing x0: notes/ch11.md, "Unclear in the PDF"',
  },
  'ch11/11.2/q-ex2-64': {
    kind: 'numeric',
    id: 'ch11/11.2/q-ex2-64',
    prompt: (
      <>
        A 256-thread block runs Fig. 11.3 on an AMD device with 64-wide wavefronts. In how many iterations of the loop is
        some wavefront divergent?
      </>
    ),
    answer: divergence(256, 64).iterations,
    unit: 'iterations',
    explain: (
      <p>
        Wavefront 0 is split by <code>threadIdx.x &gt;= stride</code> while the stride is below 64: strides 1, 2, 4, 8,
        16 and 32, six iterations. From stride 64 on, whole wavefronts drop out together. With the book's 32-wide warps it
        is five iterations, the strides up to 16 that exercise 2 names.
      </p>
    ),
    source: 'ch. 11 exercise 2, p. 286; wavefront width: notes/today/ch05-facts.md B5, ch10-facts B2',
  },
  'ch11/11.2/q-ex3': {
    kind: 'numeric',
    id: 'ch11/11.2/q-ex3',
    prompt: <>Exercise 3: how many additions does the kernel of Fig. 11.3 perform on 2048 elements?</>,
    answer: totalAdditions(2048),
    unit: 'additions',
    explain: (
      <p>
        As one 2048-element segment, p. 266's formula gives 2048·11 − 2047 = 20,481. A real launch cannot do that: one
        block holds at most 1024 threads, so the input becomes two blocks of 1024, which do 2·(1024·10 − 1023) = 18,434
        additions (and leave two separate scans for section 11.9 to stitch). The exercise does not say which it means.
      </p>
    ),
    source: 'ch. 11 exercise 3, p. 286; p. 266; the two readings: notes/ch11.md, exercises',
  },
  'ch11/11.2/q-why-not-reduction': {
    kind: 'choice',
    id: 'ch11/11.2/q-why-not-reduction',
    prompt: <>Why did chapter 10's in-place reduction need one barrier per iteration, while this scan needs two?</>,
    options: [
      {
        label: 'Unlike the reduction, in Kogge-Stone an active thread reads a slot another active thread overwrites in the same iteration.',
        correct: true,
      },
      {
        label: 'The reduction kept its partial sums in registers between iterations; the scan keeps them in shared memory.',
        why: 'Neither keeps its partial sums in registers: both work in place in an array every thread reads (shared memory here, global memory in Figs. 10.6 and 10.7).',
      },
      {
        label: 'The scan keeps all of its threads active in every iteration, where the reduction halves them each time.',
        why: 'Kogge-Stone switches off the threads below the stride too. What matters is which slots the active threads share.',
      },
      {
        label: 'The reduction needed two as well; chapter 10 left one out because that race rarely shows.',
        why: 'It was right: check which slots its active threads read and write.',
      },
    ],
    explain: (
      <p>
        In the convergent reduction each active thread reads its own slot and one <i>stride</i> to the right, and no
        active thread writes those right-hand slots in that iteration. In Kogge-Stone thread <i>i</i> reads slot{' '}
        <i>i</i> − stride, which thread <i>i</i> − stride is about to overwrite: many reduction trees run at once and use
        each other's partial sums. That reuse is what makes it fast, and what makes the second barrier necessary.
      </p>
    ),
    source: 'ch. 11 §11.2, pp. 257–258; ch. 10 Figs. 10.6 and 10.7',
  },
};

// ---------------------------------------------------------------- the section
export function S112() {
  const file = useLabFile('ch11/snippets/s112_kogge_stone.cu');
  const fn = file ? functionSource(file, 'kogge_stone_scan_kernel') : null;
  const lines = kernelSource(FIG_11_3).line;
  const q = questionsS112;
  return (
    <>
      <p>
        The obvious way to scan in parallel is one thread per output, each adding up its own prefix with a loop. Every
        output is computed at once, but the last thread still adds N values in a row, so the scan takes as long as the
        sequential loop, and the threads together do N(N − 1)/2 additions instead of N − 1. More hardware, no speedup.
      </p>
      <Source section="11.2" pages="254" />
      <Ask q={q['ch11/11.2/q-naive-work']} label="Work it out" />
      <p>
        A better start is chapter 10's idea: compute each output with a reduction tree, in log₂ steps. That needs an
        associative operator (<a href={href('ch10', '10.2')}>section 10.2</a>), and commutative too if the tree reorders
        operands. But a separate tree for output <i>i</i> still does <i>i</i> additions, so the total stays quadratic, unless
        the trees share their partial sums.
      </p>
      <Source section="11.2" pages="254–255" />

      <h2>Sharing partial sums: Kogge-Stone</h2>
      <p>
        <b>Kogge-Stone</b> is such a sharing scheme, invented in the 1970s for fast adder circuits. It works in place: the
        array starts as the inputs and turns into the outputs. In step <i>k</i> (counting from 0), every position <i>i</i>{' '}
        at least 2<sup>k</sup> from the start adds the value 2<sup>k</sup> to its left, all of them at once. After{' '}
        <i>k</i> steps position <i>i</i> holds the sum of up to 2<sup>k</sup> inputs ending at <i>i</i>, and positions 0
        to 2<sup>k</sup> − 1 are final.
      </p>
      <Source section="11.2" pages="255–256" figs="Fig. 11.2" />
      <StrideStepper />
      <Ask q={q['ch11/11.2/q-ex1']} label="Work it out" />
      <Ask q={q['ch11/11.2/q-final']} />
      <p>
        Pick y15 in the tree selector on the sixteen values: its 15 additions are exactly a reduction tree over x0 to
        x15, the heavy lines of Fig. 11.2. The other outputs' trees run at the same time and share those boxes. That
        sharing brings the total down to N·log₂N − (N − 1) additions, 49 for 16 elements. The scene below shows the
        growth, and what goes wrong when the timing slips.
      </p>
      <NoteGroup>
        <Aside>
          <p>
            The book's prose counts iterations from 1 ("iteration 2" has stride 2), its figures from 0 ("Step 1 (stride =
            2)"). These pages count from 0, as the code does, and name a step by its stride when it matters.
          </p>
        </Aside>
        <Simplification>
          <p>
            The scene draws the stepper's model: every position of a step reads the values from before it, in lockstep.
            Its race is one schedule, the book's, with barrier 2 removed; no warps, no timing.
          </p>
        </Simplification>
      </NoteGroup>
      <KsDoublingScene />

      <h2>One block, one segment</h2>
      <p>
        Fig. 11.3 is the first kernel. Each block scans its own segment of <code>SEG_SIZE</code> elements with exactly{' '}
        <code>SEG_SIZE</code> threads, one per element, in a shared buffer; stitching the segments into one scan waits
        until section 11.9. Out-of-range threads load 0, the identity, so they change nothing. Each thread keeps its
        element evolving through the loop, then writes it out.
      </p>
      <Source section="11.2" pages="256–257" figs="Fig. 11.3" />
      {fn ? (
        <Code
          tabs={[{ label: 'CUDA', code: fn.text, status: 'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware' }]}
          highlight={[...(lines.barrier1 ?? []), ...(lines.barrier2 ?? [])]}
          caption={
            <>
              Fig. 11.3 from labs/ch11/snippets/s112_kogge_stone.cu, numbered as on the page. Line 14 is printed{' '}
              <code>__ syncthreads();</code> in the book, with a space that would not compile; it is fixed here. The book
              calls the kernel <code>scan_kernel</code>.
            </>
          }
        />
      ) : null}
      <p>
        The loop has two barriers, and they do different jobs. The first, line 10, makes every write of the previous
        iteration visible before anyone reads: the same barrier as in the reduction kernels of chapter 10. The second,
        line 14, makes every read of this iteration finish before anyone overwrites: that is why each thread adds into a
        private <code>temp</code> first and stores it only after the barrier.
      </p>
      <Source section="11.2" pages="257" />
      <p>
        HIP accepts <code>__shared__</code> and <code>__syncthreads()</code> as written, and the course's header maps the
        runtime calls, so the same kernel should build for HIP (unverified: it has not been built or run on AMD hardware).
        On Metal the buffer is a <code>threadgroup</code> array declared in the kernel and
        the barrier is <code>threadgroup_barrier(mem_flags::mem_threadgroup)</code>; lab 11.1 has both versions.
      </p>
      <Today checked="2026-11-20 (ch05-facts C1, C2; ch11-facts C2)">
        <p>
          MSL allows <code>threadgroup</code> variables only in kernel functions; a helper takes a{' '}
          <code>threadgroup</code> pointer instead.
        </p>
      </Today>

      <h2>Divergence is modest</h2>
      <p>
        The test <code>threadIdx.x &gt;= stride</code> switches threads off from the front of the block. While the stride
        is smaller than a warp, the threads switched off are some of warp 0's lanes, so only warp 0 runs both paths. Once
        the stride reaches the warp width, whole warps drop out together and nothing diverges. In a block of many warps
        that is hardly felt.
      </p>
      <Source section="11.2" pages="257" />
      <p>
        Exercise 2 asks you to show it for warps of 32. Check it in the explorer, then set the width to 64, the
        wavefront of AMD's gfx9 devices, and count again.
      </p>
      <Simplification>
        <p>
          The explorer says which lanes are active, not what a divergent warp costs (ch04-facts A9, B7, C4). Metal's
          SIMD-group width is read from the pipeline, not documented (ch04-facts C2).
        </p>
      </Simplification>
      <WarpExplorer />
      <Ask q={q['ch11/11.2/q-ex2-64']} label="Work it out" />

      <h2>Why the scan needs barrier 2</h2>
      <p>
        Chapter 10's reduction got by with one barrier per iteration (<a href={href('ch10', '10.3')}>section 10.3</a>).
        There, the slots an active thread reads at distance <i>stride</i> are never written by another active thread in
        the same iteration. In Kogge-Stone they are. Thread <i>i</i> reads slot <i>i</i> − stride while thread <i>i</i> −
        stride is computing a new value for it, and if that thread stores first, thread <i>i</i> adds the wrong thing.
      </p>
      <Source section="11.2" pages="257–258" />
      <p>
        The book's example: in the stride-2 iteration, thread 6 must add the old <code>buffer_s[4]</code>, x3 + x4, to its
        own x5 + x6. If thread 4 stores x1 + … + x4 first, thread 6 stores x1 + … + x6, and the stride-4 step adds x0 + x1
        + x2 on top, so x1 and x2 are counted twice. If thread 6 happens to read first, the answer is right. Whether the
        kernel is correct depends on timing, and can change from run to run; that is what makes such bugs hard to find.
      </p>
      <Aside>
        <p>
          The book prints the corrupted value as 2x1 + 2x2 + x3 + x4 + x5 + x6. Working it through, the stride-4 step also
          adds x0, so it should read x0 + 2x1 + 2x2 + x3 + x4 + x5 + x6 (notes, "Unclear in the PDF"). The model below
          shows the full expression.
        </p>
      </Aside>
      <p>
        The hunt below runs Fig. 11.3 thread by thread on one block of 8 threads. Barrier 2 starts removed and the order
        is the book's: thread 4 runs ahead in the stride-2 step. Every value carries the inputs it counts, so you can see
        x1 and x2 arrive twice.
      </p>
      <Simplification>
        <p>
          A thread advances one source-level operation at a time, with no warps and no timing; the add straight into{' '}
          <code>buffer_s</code> is split into its loads and its store. Uninitialised shared memory is NaN. "Lockstep" is
          an idealised single warp.
        </p>
      </Simplification>
      <RaceHunt />
      <p>
        Three things to take from the hunt. With barrier 2 removed, the book's order fails exactly as described: only
        position 6 is wrong. "Last thread first" happens to be right, because every read comes before the store that
        would spoil it. And lockstep hides every one of these bugs, which is why a test on a GPU that passes proves little.
        Barrier 1 guards a different dependence: remove it instead, and the reversed order reads slots nobody has written
        yet.
      </p>
      <Outcomes />
      <Ask q={q['ch11/11.2/q-race']} />
      <Ask q={q['ch11/11.2/q-why-not-reduction']} />
      <p>
        The root cause is the sharing itself. Kogge-Stone runs many reduction trees at once and lets them reuse each
        other's partial sums, so the value one thread is about to replace is an input to another thread's tree. Section
        11.5 counts what that sharing saves, and section 11.3 removes barrier 2 by a different route.
      </p>
      <Source section="11.2" pages="258" />
      <Ask q={q['ch11/11.2/q-ex3']} label="Work it out" />

      <h2>The same race in production code</h2>
      <p>
        PyTorch's scan kernel for Apple GPUs carries a running total from one chunk of a long row to the next through a
        small shared array. Its source has a barrier right after every thread reads that carry, with a comment that names
        this exact bug: without it, the next chunk's writes can overwrite the carry before everyone has read it. (The
        example is ours; the book does not look at real backends.) Lab 11.3 hunts the same kind of race in scan kernels
        on the emulator.
      </p>
      <Today checked="2026-11-20 (ch11-facts D4)">
        <p>
          <code>ScanKernel.metal</code>: "Ensure every thread has read the carry before the next iteration overwrites
          simdgroup_sums; otherwise long (multi-iteration) scans race."
        </p>
      </Today>
      <p>
        Fig. 11.3 pays {iterations(1024) * 2} barriers for one block of 1024 elements, two in each of its{' '}
        {iterations(1024)} iterations. Section <a href={href('ch11', '11.3')}>11.3</a> halves that.
      </p>
    </>
  );
}
