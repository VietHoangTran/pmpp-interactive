import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  BUILDER_OPS,
  DIST_LABEL,
  FIG103A,
  FIG103B,
  PAIR_RULES,
  SHAPES,
  SHAPE_LABEL,
  builderResult,
  builderTree,
  compareOrders,
  f32hex,
  f32short,
  f64short,
  generate,
  orderIndependent,
  surveyOrders,
  tieRow,
  tournament,
  treeStats,
  winners,
  type BuilderOpId,
  type Dist,
  type PairRule,
  type Shape,
} from '../../sim/reducetree';
import { TreeFoldScene, TreeSvg } from './TreeFoldScene';

import { NoteGroup } from './NoteGroup';
/** A caption above a group of buttons (a <label> would forward clicks on the caption to the first button). */
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
/** Long button groups wrap on a narrow screen instead of running off the bench. */
const WRAP: CSSProperties = { flexWrap: 'wrap' };
const fmt = (x: number) => String(Number(x.toFixed(2)));
const SHAPE_NOTE: Record<Shape, string> = {
  loop: 'Fig. 10.2: one element at a time, from element 0',
  neighbours: 'Fig. 10.3: adjacent pairs at every level',
  halves: 'node j with node j + m/2, the order of section 10.4. The leaves are drawn in the order the tree combines them',
};

function ShapePicker({ shape, set }: { shape: Shape; set: (s: Shape) => void }) {
  return (
    <span style={FIELD}>
      Order
      <span className="seg" role="group" aria-label="Order">
        {SHAPES.map((s) => (
          <button key={s} aria-pressed={shape === s} onClick={() => set(s)}>
            {SHAPE_LABEL[s]}
          </button>
        ))}
      </span>
    </span>
  );
}

/** Operations in flight per step, as bars. */
function StepBars({ ops, label, max }: { ops: number[]; label: string; max?: number }) {
  const top = max ?? Math.max(...ops);
  const bw = Math.max(6, Math.min(26, 260 / ops.length));
  const H = 64;
  return (
    <svg viewBox={`0 0 ${ops.length * (bw + 3) + 4} ${H + 16}`} role="img" aria-label={label} data-part="bars" data-ops={ops.join(' ')}
      style={{ width: ops.length * (bw + 3) + 4, maxWidth: '100%', display: 'block', marginTop: '0.4rem' }}>
      {ops.map((k, s) => {
        const h = Math.max(1, (k / top) * H);
        return (
          <g key={s}>
            <rect x={2 + s * (bw + 3)} y={H - h} width={bw} height={h} fill="var(--ink-2)" />
            {ops.length <= 16 ? (
              <text x={2 + s * (bw + 3) + bw / 2} y={H + 12} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{s + 1}</text>
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------- interactive 1: the tree builder
const POOL = [...FIG103B, ...FIG103A];

function LeafInput({ value, onValue, k }: { value: number; onValue: (v: number) => void; k: number }) {
  const [text, setText] = useState(String(value));
  useEffect(() => {
    if (Number(text) !== value) setText(String(value));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);
  return (
    <input
      type="number"
      step="1"
      value={text}
      aria-label={`leaf ${k}`}
      style={{ width: '3.6rem', padding: '0.25rem 0.3rem' }}
      onChange={(e) => {
        setText(e.target.value);
        const v = Number(e.target.value);
        if (e.target.value.trim() !== '' && Number.isFinite(v)) onValue(Math.max(-999, Math.min(999, Math.round(v))));
      }}
    />
  );
}

function TreeBuilder() {
  const [leaves, setLeaves] = useState<number[]>(FIG103B);
  const [op, setOp] = useState<BuilderOpId>('sum');
  const [shape, setShape] = useState<Shape>('neighbours');
  const [logN, setLogN] = useState(10);
  const n = leaves.length;
  const tree = useMemo(() => builderTree(leaves, op, shape), [leaves, op, shape]);
  const loop = useMemo(() => builderResult(leaves, op, 'loop'), [leaves, op]);
  const mine = useMemo(() => builderResult(leaves, op, shape), [leaves, op, shape]);
  const same = mine === loop;
  const stats = treeStats(2 ** logN);
  const resize = (k: number) => setLeaves(Array.from({ length: k }, (_, j) => leaves[j] ?? POOL[j]));
  const dx = n > 8 ? 42 : 56;
  // Rows far enough apart that a leaf's index label clears the first row of operations.
  const dy = shape === 'loop' ? (n > 8 ? 36 : 40) : n > 8 ? 54 : 62;
  const margin = 36;
  const width = 2 * margin + (n - 1) * dx + (shape === 'loop' ? 30 : 0);
  const height = 24 + (tree.levels.length - 1) * dy + 30;
  const peak = Math.max(...tree.opsPerStep);

  return (
    <Bench
      title="Build a reduction tree"
      task="Edit the leaves, pick an operator and an order. Which operators give the loop's answer in every order?"
      foot={
        <>
          Integers are combined exactly, so any disagreement with the loop comes from the order alone. "join" writes its
          two operands side by side: it shows the order in which the leaves were combined.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Leaves
          <span className="seg" role="group" aria-label="Leaves">
            {[4, 8, 16].map((k) => (
              <button key={k} aria-pressed={n === k} onClick={() => resize(k)}>
                {k}
              </button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          Operator
          <span className="seg" role="group" aria-label="Operator" style={WRAP}>
            {BUILDER_OPS.map((o) => (
              <button key={o.id} aria-pressed={op === o.id} onClick={() => setOp(o.id)}>
                {o.label}
              </button>
            ))}
          </span>
        </span>
        <ShapePicker shape={shape} set={setShape} />
        <span style={FIELD}>
          Book examples
          <span className="seg" role="group" aria-label="Book examples" style={WRAP}>
            <button onClick={() => { setLeaves(FIG103B); setOp('sum'); setShape('neighbours'); }}>Fig. 10.3(b), sum</button>
            <button onClick={() => { setLeaves(FIG103A); setOp('max'); setShape('neighbours'); }}>Fig. 10.3(a), max</button>
          </span>
        </span>
      </div>
      <div className="row" role="group" aria-label="Leaf values" style={{ gap: '0.3rem', marginBottom: '0.6rem' }}>
        {leaves.map((v, k) => (
          <LeafInput key={k} k={k} value={v} onValue={(nv) => setLeaves(leaves.map((x, j) => (j === k ? nv : x)))} />
        ))}
      </div>
      <p className="muted" style={{ margin: '0 0 0.4rem' }}>
        {SHAPE_LABEL[shape]}: {SHAPE_NOTE[shape]}. Time flows down, one row per step.
      </p>
      <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="The reduction tree" data-part="builder-tree"
          style={{ width: '100%', minWidth: Math.min(width, n * 30), maxWidth: width * 1.15, display: 'block' }}>
          <TreeSvg tree={tree} x={margin} y={6} dx={dx} dy={dy} w={n > 8 ? 34 : 40} maxChars={8} name="builder"
            under={(k) => `x${k}`} operandOrder={shape === 'halves'} />
        </svg>
      </div>
      <div className="cols" style={{ marginTop: '0.8rem' }}>
        <div className="panel" aria-live="polite" data-readout="result" data-same={same ? '1' : '0'}>
          <p style={{ marginTop: 0, overflowWrap: 'anywhere' }}>
            <code data-part="expr">{tree.expr}</code>
          </p>
          <p style={{ marginBottom: 0, overflowWrap: 'anywhere' }}>
            This order: <b data-value="mine">{tree.root.value}</b>. The loop: <b data-value="loop">{loopDisplay(leaves, op)}</b>.{' '}
            {same ? <span className="chip ok">same as the loop</span> : <span className="chip fail">differs from the loop</span>}
          </p>
        </div>
        <div className="panel" data-readout="counters">
          <p style={{ margin: 0 }}>
            Work <b data-counter="work">{tree.work}</b> operations. Span <b data-counter="span">{tree.span}</b> steps. At
            most <b data-counter="peak">{peak}</b> at once.
          </p>
          <StepBars ops={tree.opsPerStep} label="Operations in each step" />
          <p className="muted" style={{ margin: '0.2rem 0 0', fontSize: '0.85em' }}>operations in each step</p>
        </div>
      </div>
      <div className="panel" style={{ marginTop: '1rem' }} data-readout="scale">
        <label className="field" style={{ width: '100%' }}>
          <span>
            Counters only, for a tree over N = <b data-stat="n">{stats.n}</b> inputs
          </span>
          <input type="range" min={1} max={10} value={logN} aria-label="N for the counters" onChange={(e) => setLogN(Number(e.target.value))} style={{ width: '100%', maxWidth: '24rem' }} />
        </label>
        <p style={{ margin: '0.5rem 0 0' }}>
          Work N − 1 = <b data-stat="work">{stats.work}</b>. Span log₂ N = <b data-stat="span">{stats.span}</b> steps.
          Peak N/2 = <b data-stat="peak">{stats.peak}</b> operations at once. Average (N − 1)/log₂ N ={' '}
          <b data-stat="average">{fmt(stats.average)}</b>.
        </p>
        <StepBars ops={stats.opsPerStep} label="Operations in each step for N inputs" />
      </div>
    </Bench>
  );
}
const loopDisplay = (leaves: number[], op: BuilderOpId) => builderTree(leaves, op, 'loop').root.value;

// ---------------------------------------------------------------- interactive 2: three orders, one float32 sum
const DISTS: Dist[] = ['uniform', 'softmax', 'wide'];
const trialsFor = (n: number) => (n >= 16384 ? 10 : n >= 4096 ? 30 : 100);
const DEFAULT_ORDERS = compareOrders(generate(1024, 'softmax', 1));
const DEFAULT_SURVEY = surveyOrders(1024, 'softmax', trialsFor(1024));

function OrdersBench() {
  const [logN, setLogN] = useState(10);
  const [dist, setDist] = useState<Dist>('softmax');
  const [seed, setSeed] = useState(1);
  const n = 2 ** logN;
  const values = useMemo(() => generate(n, dist, seed), [n, dist, seed]);
  const r = useMemo(() => compareOrders(values), [values]);
  const trials = trialsFor(n);
  const survey = useMemo(() => surveyOrders(n, dist, trials), [n, dist, trials]);
  return (
    <Bench
      title="Three orders, one float32 sum"
      task="Change the size, the kind of data and the seed. When do the orders agree, and which is closest to the exact sum?"
      foot={
        <>
          Values are generated as float32. Each order adds them in float32, rounding after every addition. The reference
          is the same values added in float64 with compensated summation, far more accurate than any float32 order. ULP
          distance counts the float32 values between a sum and the float32 nearest the reference.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <span>
            N = <b data-value="n">{n}</b> values
          </span>
          <input type="range" min={3} max={16} value={logN} aria-label="Number of values" onChange={(e) => setLogN(Number(e.target.value))} style={{ width: '14rem' }} />
        </label>
        <span style={FIELD}>
          Data
          <span className="seg" role="group" aria-label="Data" style={WRAP}>
            {DISTS.map((d) => (
              <button key={d} aria-pressed={dist === d} onClick={() => setDist(d)}>
                {DIST_LABEL[d]}
              </button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          Seed {seed}
          <button onClick={() => setSeed(seed + 1)}>New data</button>
        </span>
      </div>
      <p className="muted" style={{ marginTop: 0, overflowWrap: 'anywhere' }}>
        First values: {values.slice(0, 6).map(f32short).join(', ')}, …
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" data-part="orders">
          <thead>
            <tr>
              <th>Order</th>
              <th className="num">float32 sum</th>
              <th className="num">bits</th>
              <th className="num">ULPs from the reference</th>
            </tr>
          </thead>
          <tbody>
            {SHAPES.map((s) => (
              <tr key={s} data-order={s} data-hex={f32hex(r.sums[s])} data-ulps={r.ulps[s]}>
                <td>
                  {SHAPE_LABEL[s]}
                  {r.closest.includes(s) ? <span className="chip ok" style={{ marginLeft: 6 }}>closest</span> : null}
                </td>
                <td className="num">{f32short(r.sums[s])}</td>
                <td className="num mono">{f32hex(r.sums[s])}</td>
                <td className="num">{r.ulps[s]}</td>
              </tr>
            ))}
            <tr data-order="reference">
              <td>float64 reference</td>
              <td className="num">{f64short(r.ref)}</td>
              <td className="num muted">nearest float32 {f32hex(r.ref)}</td>
              <td className="num">0</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p aria-live="polite" data-readout="distinct" data-distinct={r.distinct} style={{ marginBottom: '0.4rem' }}>
        <b>{r.distinct === 1 ? 'One result:' : `${r.distinct} different results.`}</b>{' '}
        {r.distinct === 1 ? 'all three orders agree, bit for bit, on this data.' : 'The same numbers, added in different groupings.'}
      </p>
      <p className="muted" style={{ marginTop: 0 }} data-readout="survey">
        Over {survey.trials} data sets of this kind and size (seeds 1 to {survey.trials}): the orders disagree in{' '}
        <b>{survey.disagree}</b>. Closest to the reference (ties count for each): the loop in {survey.closest.loop},
        neighbours in {survey.closest.neighbours}, halves in {survey.closest.halves}.
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive 3: who wins a tie
const ARG_RULES: PairRule[] = ['lower', 'left', 'higher', 'nanfirst'];
const MAX_RULES: PairRule[] = ['fmax', 'cmp'];

function TieBench() {
  const [n, setN] = useState(8);
  const [maxima, setMaxima] = useState<number[]>([1, 4]);
  const [nanOn, setNanOn] = useState(false);
  const [nanAt, setNanAt] = useState(6);
  const [rule, setRule] = useState<PairRule>('left');
  const row = useMemo(() => tieRow(n, maxima, nanOn ? nanAt : null), [n, maxima, nanOn, nanAt]);
  const w = useMemo(() => winners(row, rule), [row, rule]);
  const indep = orderIndependent(row, rule);
  const info = PAIR_RULES[rule];
  const resize = (k: number) => {
    setN(k);
    const m = maxima.filter((x) => x < k);
    setMaxima(m.length === 2 ? m : [1, 4]);
    if (nanAt >= k) setNanAt(k - 2);
  };
  const show = (v: number) => (Number.isNaN(v) ? 'NaN' : f32short(v));
  const radio = (r: PairRule) => (
    <label key={r} className="check" style={{ display: 'flex' }}>
      <input type="radio" name="tie-rule" checked={rule === r} onChange={() => setRule(r)} /> {PAIR_RULES[r].label}
    </label>
  );
  return (
    <Bench
      title="Who wins a tie"
      task="Pick a combine rule and compare the winner under the three orders. Then move a maximum, or add a NaN."
      foot={
        <>
          Each order is one tree over (value, index) pairs, combined as combine(left, right). Click a cell to move the
          older of the two maxima there.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Logits
          <span className="seg" role="group" aria-label="Logits">
            {[8, 16].map((k) => (
              <button key={k} aria-pressed={n === k} onClick={() => resize(k)}>
                {k}
              </button>
            ))}
          </span>
        </span>
        <label className="check">
          <input type="checkbox" checked={nanOn} onChange={(e) => setNanOn(e.target.checked)} /> put a NaN at index
        </label>
        <select value={nanAt} aria-label="NaN index" disabled={!nanOn} onChange={(e) => setNanAt(Number(e.target.value))}>
          {Array.from({ length: n }, (_, k) => (
            <option key={k} value={k}>
              {k}
            </option>
          ))}
        </select>
      </div>
      <div role="group" aria-label="Row of logits" style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: '0.9rem' }}>
        {row.map((v, k) => {
          const isMax = maxima.includes(k) && !Number.isNaN(v);
          return (
            <button
              key={k}
              data-cell={k}
              data-max={isMax ? 1 : 0}
              aria-label={`logit ${k}: ${show(v)}`}
              onClick={() => !maxima.includes(k) && setMaxima([maxima[1], k])}
              style={{
                width: 52,
                padding: '0.2rem 0',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                fontVariantNumeric: 'tabular-nums',
                border: isMax ? '2px solid var(--ink)' : '1px solid var(--rule)',
                background: 'var(--cell)',
                color: 'var(--ink)',
                fontWeight: isMax ? 700 : 450,
              }}
            >
              <span>{show(v)}</span>
              <span className="muted" style={{ fontSize: '0.75em' }}>#{k}</span>
            </button>
          );
        })}
      </div>
      <div className="cols">
        <div role="radiogroup" aria-label="Combine rule">
          <p style={{ margin: '0 0 0.3rem' }} className="muted">argmax: the index matters</p>
          {ARG_RULES.map(radio)}
          <p style={{ margin: '0.6rem 0 0.3rem' }} className="muted">plain max: only the value matters</p>
          {MAX_RULES.map(radio)}
        </div>
        <div className="panel" aria-live="polite" data-readout="winners" data-independent={indep ? '1' : '0'}>
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              {SHAPES.map((s) => (
                <tr key={s} data-order={s} data-index={w[s].i} data-value={show(w[s].v)}>
                  <td>{SHAPE_LABEL[s]}</td>
                  <td className="num">{info.argmax ? `index ${w[s].i} (value ${show(w[s].v)})` : `value ${show(w[s].v)}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ margin: '0.6rem 0 0' }}>
            {indep ? <span className="chip ok">the same in every order</span> : <span className="chip fail">depends on the order</span>}
          </p>
          <p className="muted" style={{ margin: '0.4rem 0 0' }} data-readout="rule-note">
            {info.note}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const AVG4096 = treeStats(4096);
const VENUES = tournament(8, 2);

export const questionsS102: Record<string, Question> = {
  'ch10/10.2/q-avg-4096': {
    kind: 'numeric',
    id: 'ch10/10.2/q-avg-4096',
    prompt: <>A reduction tree over 4096 elements. What is its average parallelism: operations per step, on average?</>,
    answer: AVG4096.average,
    tol: 0.01,
    unit: 'operations per step',
    explain: (
      <p>
        4095 operations in log₂ 4096 = 12 steps: {AVG4096.average}. The first step runs {AVG4096.peak} operations at
        once. Hardware sized for that peak is used, on average, at a sixth of it.
      </p>
    ),
    source: 'ch. 10 §10.2, pp. 225–226',
  },
  'ch10/10.2/q-subtract': {
    kind: 'choice',
    id: 'ch10/10.2/q-subtract',
    prompt: <>Can a parallel tree compute x0 − x1 − x2 − … − x7, with the answer the left-to-right loop gives?</>,
    options: [
      { label: 'No. Subtraction is not associative, so no tree can compute it.', why: 'Change the operator, not the tree: the same quantity is a sum.' },
      { label: 'Yes, after rewriting it as x0 + (−x1) + … + (−x7) and summing with any tree.', correct: true },
      { label: 'Yes, any tree of subtractions gives the same answer, as long as the leaves stay in order.', why: "Try it in the builder: on Fig. 10.3(b)'s leaves the neighbours tree gives −2 and the loop −36." },
    ],
    explain: (
      <p>
        The tree needs an associative operator, and subtraction is not one: (1 − 2) − 3 ≠ 1 − (2 − 3). But subtracting
        is adding the negation, and addition is associative. Negate every element but the first, then sum in any order.
      </p>
    ),
    source: 'ch. 10 §10.2, p. 224',
  },
  'ch10/10.2/q-commute': {
    kind: 'choice',
    id: 'ch10/10.2/q-commute',
    prompt: <>Which order needs a commutative operator, not merely an associative one?</>,
    options: [
      { label: 'neighbours: ((x0 x1)(x2 x3))((x4 x5)(x6 x7)), adjacent leaves paired first', why: 'The leaves stay in their original sequence. Only the parentheses moved.' },
      { label: 'halves: x0 with x4, x1 with x5, then pairs of those', correct: true },
      { label: 'the loop, one element at a time', why: 'The loop is the reference order: it neither regroups nor reorders.' },
    ],
    explain: (
      <p>
        Read the leaves of the halves tree from left to right: x0, x4, x2, x6, x1, x5, x3, x7. The operands are no
        longer in their original sequence, so the result equals the loop's only if swapping operands changes nothing. The
        "join" operator in the builder shows it: associative but not commutative, it agrees with the loop under
        neighbours and not under halves. Section 10.4's kernel uses the halves order.
      </p>
    ),
    source: 'ch. 10 §10.2, p. 225; §10.4, pp. 232–233',
  },
  'ch10/10.2/q-venues': {
    kind: 'numeric',
    id: 'ch10/10.2/q-venues',
    prompt: (
      <>
        A knockout tournament with 8 teams has only 2 venues, each able to host one game a day, and a round must finish
        before the next begins. How many days does it take?
      </>
    ),
    answer: VENUES.days,
    unit: 'days',
    explain: (
      <p>
        Quarter-finals: 4 games on 2 venues, 2 days. Semi-finals: 1 day. Final: 1 day. Total {VENUES.days}, against 3
        with 4 venues. Sharing a resource between operations lengthens the reduction; section 10.10 makes the same
        trade on purpose.
      </p>
    ),
    source: 'ch. 10 §10.2 sidebar, pp. 226–227 (the numbers are ours)',
  },
  'ch10/10.2/q-argmax-rule': {
    kind: 'choice',
    id: 'ch10/10.2/q-argmax-rule',
    prompt: <>A GPU argmax combines (value, index) pairs in a tree. Which combine rule makes its answer independent of the order the tree uses?</>,
    options: [
      { label: 'The larger value wins; equal values go to the lower index.', correct: true },
      { label: 'The larger value wins; on equal values keep the left operand.', why: 'Which operand is on the left depends on the tree. The explorer shows the winner moving.' },
      { label: 'The right operand replaces the left only if its value is strictly greater.', why: 'The same rule as keeping the left operand, in other words.' },
      { label: 'Compare indices only when both values are NaN.', why: 'That settles a corner and leaves ordinary ties to whichever operand is on the left.' },
    ],
    explain: (
      <p>
        Ordering pairs by value and then by lower index is a total order on finite values, and the maximum of a total
        order is associative and commutative: every tree gives the same pair, the first maximal index. That is also the
        index PyTorch documents for <code>torch.argmax</code>. A NaN needs a rule of its own on top.
      </p>
    ),
    source: 'ch. 10 §10.2, pp. 224–225; PyTorch: notes/today/ch10-facts.md D4',
  },
};

// ---------------------------------------------------------------- the section
export function S102() {
  const q = questionsS102;
  return (
    <>
      <p>
        The loop of section 10.1 combines one element at a time, so N elements take N steps however many processors you
        have. A parallel reduction combines pairs instead. In the first step, separate operations each combine two
        inputs. In the next, half as many combine those results, and so on until one value is left. Drawn with time
        flowing downward and one row per step, this is a <b>reduction tree</b>. Eight inputs take 4 operations, then 2,
        then 1.
      </p>
      <Source section="10.2" pages="223–224" figs="Fig. 10.3" />
      <p>
        The book draws two: a max over 3, 1, 7, 0, 4, 1, 6, 3, which gives 7, and a sum over 4, 7, 2, 3, 8, 5, 9, 6,
        which gives 44. The second is the one the rest of the chapter's kernels compute. Despite the name, a reduction
        tree is not a data structure. Its edges are values handed from one step to the next.
      </p>
      <NoteGroup>
        <Source section="10.2" pages="225, 227" />
        <Simplification>
          <p>
            The builder uses exact integer arithmetic, so only the order can change a result. Every operation takes one
            step; nothing models threads, memory or how long a step takes.
          </p>
        </Simplification>
      </NoteGroup>
      <TreeBuilder />
      <p>
        The animation runs the loop and the tree on the book's eight values and then looks at what changes when there
        are many more of them.
      </p>
      <Simplification>
        <p>
          One adder per operation and one step per level. The float sums round to float32 after every addition, with no
          fused operations. Nothing models threads, memory or how long a step takes.
        </p>
      </Simplification>
      <TreeFoldScene />

      <h2>Same work, shorter span</h2>
      <p>
        Two numbers describe a parallel algorithm. Its <b>work</b> is how many operations it performs. Its <b>span</b>{' '}
        is how long it takes if resources are unlimited, counted in steps, where a step is a period in which some or all
        threads each perform one operation. The vector addition of chapter 2 does N operations in one step. A sequential
        algorithm's work and span are the same number.
      </p>
      <Source section="10.2" pages="225–226" />
      <p>
        A tree over N inputs does N/2 + N/4 + … + 1 = N − 1 operations: the same work as the loop. Its span is log₂ N
        steps. For 1024 inputs that is 10 steps, with 512 operations in the first and one in the last. Divide the work
        by the span and you get the <b>average parallelism</b>: 1023 / 10 = 102.3, against a peak of 512. A machine big
        enough for the first step sits mostly idle for the rest. That uneven demand is what makes reduction trees
        awkward for parallel hardware, and most of this chapter's kernels are ways of living with it.
      </p>
      <Source section="10.2" pages="226" />
      <p>
        A tree that does no more work than the loop is not a given. Chapter 11 meets parallel algorithms that do more
        work than the sequential version, and the question of how much more is acceptable.
      </p>
      <Ask q={q['ch10/10.2/q-avg-4096']} />

      <h2>A tournament is a max tree</h2>
      <p>
        A knockout tournament is a reduction tree whose operator returns the team that wins. Eight teams leave four
        winners after the quarter-finals, two after the semi-finals, one after the final: three rounds, log₂ 8. 1024
        teams would need 10 rounds. A tree is fast, and hungry for resources: in the first
        round every game needs its own stadium and staff. Organisers who share a stadium between two games of a round
        have to play them on different days, and the tournament gets longer. Section 10.10 makes the same trade
        deliberately, giving up some parallelism to use the hardware better.
      </p>
      <NoteGroup>
        <Source section="10.2 sidebar" pages="226–227" figs="Fig. 10.4" />
        <Aside>
          <p>
            The sidebar says the tournament in Fig. 10.4 "has a span of four steps". Its eight teams play three rounds,
            so the span is three.
          </p>
        </Aside>
      </NoteGroup>
      <Ask q={q['ch10/10.2/q-venues']} />

      <h2>The tree changes the order of operations</h2>
      <p>
        The loop and the tree do not perform the same operations. On the max example, the loop compares 3 with 1, then
        the winner with 7, and so on along the list. The tree compares 7 with 0 before it has combined 3 with 1. Written
        out, the loop computes ((((((3 max 1) max 7) max 0) max 4) max 1) max 6) max 3, and the tree computes ((3 max 1)
        max (7 max 0)) max ((4 max 1) max (6 max 3)). Both give 7 because max is <b>associative</b>: (a ⊕ b) ⊕ c = a ⊕
        (b ⊕ c) for every a, b, c, so the parentheses can go anywhere.
      </p>
      <Source section="10.2" pages="224–225" />
      <p>
        Integer addition is associative: (1 + 2) + 3 = 1 + (2 + 3). In C++ that holds only while nothing overflows,
        because a different grouping can overflow where the original did not, and signed overflow is undefined
        behaviour. Integer subtraction is not associative: (1 − 2) − 3 ≠ 1 − (2 − 3). It can still be reduced in
        parallel once it is rewritten as an addition of negated operands.
      </p>
      <Source section="10.2" pages="224 (footnote 2)" />
      <Ask q={q['ch10/10.2/q-subtract']} />
      <p>
        Floating-point addition is associative in mathematics and not in a computer. Each addition rounds its result to
        the nearest representable value, and which intermediate results get rounded depends on the grouping. Appendix A
        works an example in a five-bit float format: 1 + 1 + 0.25 + 0.25 added left to right gives 2.0, because each
        0.25 is too small to change 2.0; added in pairs, (1 + 1) + (0.25 + 0.25), it gives the exact 2.5. The book
        notes that a nearby example goes the other way. Applications that accept results within a tolerance treat float
        addition as associative anyway, and so do the kernels in this chapter.
      </p>
      <NoteGroup>
        <Source section="10.2" pages="224" figs="Appendix A §A.6, pp. 593–594" />
        <Today checked="2026-11-20">
          <p>
            NVIDIA's programming guide now has its own three-value example (§5.6.1.4), and its dot-product example
            (§5.6.1.6) calls the tree-shaped order "the parallel algorithm".
          </p>
          <p className="muted">notes/today/ch10-facts.md A9</p>
        </Today>
      </NoteGroup>
      <p>How much does the order matter for the sum a softmax divides by? Guess before you look.</p>
      <Simplification>
        <p>
          Rounding to nearest after every addition: no fused operations, no flushing to zero, no reassociation by a
          compiler.
        </p>
      </Simplification>
      <Predict
        id="ch10/10.2/p-orders"
        prompt="1024 float32 values, the terms of a softmax denominator for a row with one large logit, are summed in the three orders: the loop, neighbours and halves. How many different results will the three give: 1, 2 or 3?"
        unit="different results"
      >
        {(v) => (
          <p>
            <b>
              {DEFAULT_ORDERS.distinct} on the first data set below.
            </b>{' '}
            {v === DEFAULT_ORDERS.distinct ? 'That is what you predicted. ' : ''}
            {v === 1
              ? 'One would mean the grouping never changed a rounding, across a thousand additions. It almost always does.'
              : v === 3 && DEFAULT_ORDERS.distinct === 2
                ? 'Three happens too; on this data set the two tree orders agree with each other.'
                : ''}{' '}
            Over {DEFAULT_SURVEY.trials} data sets of this kind, at least two orders disagree in {DEFAULT_SURVEY.disagree}.
            The differences sit in the last bits. Explore when they appear.
          </p>
        )}
      </Predict>
      <OrdersBench />
      <p>
        The differences are small and they are real. None of the three orders is the exact sum in general. The tree
        orders land closer to it more often than the loop does as N grows, because each value passes through only log₂ N
        additions instead of up to N, but not always. For softmax this means two correct kernels can produce
        probabilities that differ in the last bits, and a test against a CPU loop needs a tolerance, never an equality.
      </p>
      <Beyond>
        <p>The softmax framing and the "log₂ N additions per value" explanation are ours.</p>
      </Beyond>

      <h2>Reordering the operands as well</h2>
      <p>
        Section 10.4 goes one step further. Its kernel combines x0 with x4 and x1 with x5, not x0 with x1. That changes
        the sequence of the operands, not only the parentheses, so the operator must also be <b>commutative</b>: a ⊕ b =
        b ⊕ a. Max, min, addition and multiplication are; subtraction is not. Set the builder to "join", an operator that
        is associative but not commutative, and compare the neighbours order with the halves order.
      </p>
      <Source section="10.2" pages="225" />
      <Ask q={q['ch10/10.2/q-commute']} />
      <p>
        One caveat in a footnote matters more than it looks. Floating-point max and min are not exactly commutative when
        one operand is a NaN: max(NaN, 1) and max(1, NaN) need not agree. Which max you write decides what happens, and
        a tree is free to put the NaN on either side.
      </p>
      <NoteGroup>
        <Source section="10.2" pages="225 (footnote 3)" />
        <Today checked="2026-11-20 (ch10-facts A10)">
          <p>
            CUDA's <code>fmaxf</code> treats NaN "as missing data"; <code>a &gt; b ? a : b</code> returns <code>b</code>{' '}
            if either is NaN.
          </p>
        </Today>
      </NoteGroup>

      <h2>Who wins a tie</h2>
      <p>
        Greedy decoding picks the index of the largest logit. A GPU computes it as a max tree over (value, index) pairs,
        so the pair operator must be associative and commutative too. For pairs that is a real constraint: two logits
        can be equal, and a rule such as "on a tie, keep the left operand" gives a winner that depends on which operand
        the tree happens to put on the left. Find out which rules survive every order.
      </p>
      <NoteGroup>
        <Beyond>
          <p>Argmax as a pair reduction, and the library comparisons below, are ours.</p>
        </Beyond>
        <Simplification>
          <p>
            One tree per order, no warps or blocks. Library behaviours named in the explorer are our reading of their
            documentation and source, not tests on hardware.
          </p>
        </Simplification>
      </NoteGroup>
      <TieBench />
      <p>
        Libraries do not agree. PyTorch documents that <code>torch.argmax</code> returns the first maximal index, and
        CUB's <code>ArgMax</code> finds "the first device-wide maximum". In llama.cpp, by our reading of the source,
        the host's greedy sampler keeps the lowest index on a tie, the CUDA argmax kernel's answer depends on which lane
        held which index, and the Metal kernel takes the largest index. One model, run three ways, can pick different
        tokens on an exact tie. NaN is a second policy on top: PyTorch and MLX let a NaN win, CUB's <code>Max</code>{' '}
        ignores NaN, and llama.cpp's CUDA argmax skips it. Exact ties in float logits are rare; milestone M4 states its
        rule and tests it with planted ties anyway.
      </p>
      <Today checked="2026-11-20">
        <p>
          PyTorch 2.15: "If there are multiple maximal values then the indices of the first maximal value are returned."
          llama.cpp's behaviour is read from source, not observed.
        </p>
        <p className="muted">ch10-facts A13, D2, D3, D4</p>
      </Today>
      <Ask q={q['ch10/10.2/q-argmax-rule']} />
      <p>
        With the tree, its cost and its two conditions in hand, <a href={href('ch10', '10.3')}>section 10.3</a> writes
        the first kernel.
      </p>
    </>
  );
}
