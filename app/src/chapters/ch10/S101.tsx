import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { NoteGroup } from './NoteGroup';
import {
  BOOK_LIST,
  FOLD_OPS,
  HUGE_ROW,
  NEGATIVE_ROW,
  START_LABEL,
  f32short,
  f64short,
  foldSequential,
  randomRow,
  randomRowTest,
  type FoldOpId,
  type StartId,
} from '../../sim/reducetree';

/** A caption above a group of buttons (a <label> would forward clicks on the caption to the first button). */
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
/** Long button groups wrap on a narrow screen instead of running off the bench. */
const WRAP: CSSProperties = { flexWrap: 'wrap' };
const OPS: FoldOpId[] = ['sum', 'product', 'max', 'min', 'argmax'];
const STARTS: StartId[] = ['identity', 'zero', 'first', 'negmax'];
const PRESETS: { label: string; values: number[] }[] = [
  { label: "the book's list", values: BOOK_LIST },
  { label: 'logits, all negative', values: NEGATIVE_ROW },
  { label: 'one huge logit', values: HUGE_ROW },
];
const listText = (v: number[]) => v.map((x) => (Number.isInteger(x) ? x.toFixed(1) : String(x))).join(', ');
function parseList(s: string): number[] | null {
  const parts = s.split(/[\s,]+/).filter(Boolean);
  if (parts.length < 1 || parts.length > 16) return null;
  const v = parts.map((p) => Number(p.replace('−', '-')));
  return v.every((x) => Number.isFinite(x)) ? v : null;
}

// ---------------------------------------------------------------- the fold explorer
function FoldExplorer() {
  const [op, setOp] = useState<FoldOpId>('sum');
  const [start, setStart] = useState<StartId>('identity');
  const [values, setValues] = useState<number[]>(BOOK_LIST);
  const [text, setText] = useState(listText(BOOK_LIST));
  const [seed, setSeed] = useState(1);
  const trace = useMemo(() => foldSequential(values, op, start, 'f32'), [values, op, start]);
  const n = trace.steps.length;
  const [shown, setShown] = useState(n);
  const [playing, setPlaying] = useState(false);
  // Any change of list, operator or start value shows the whole run again.
  useEffect(() => {
    setShown(n);
    setPlaying(false);
  }, [values, op, start, n]);
  useEffect(() => {
    if (!playing) return;
    if (shown >= n) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => setShown((k) => k + 1), 650);
    return () => clearTimeout(t);
  }, [playing, shown, n]);
  const random = useMemo(() => randomRowTest(op, start), [op, start]);
  const bad = parseList(text) === null;
  const pick = (v: number[]) => {
    setValues(v);
    setText(listText(v));
  };
  const isArg = op === 'argmax';
  const pair = (v: number, i: number) => `(${f32short(v)}, ${i < 0 ? 'none' : i})`;
  const accText = (v: number, i: number) => (isArg ? pair(v, i) : f32short(v));
  const done = shown >= n;
  const identity = FOLD_OPS[op].identityLabel;

  return (
    <Bench
      title="One loop, any operator"
      task="Pick an operator, a list and a starting value. Run the loop and compare its answer with the right one."
      foot={
        <>
          The loop runs in float32, rounding after every operation, as a C <code>float</code> loop does. The right
          answer is computed separately in float64 with the operator's own identity; for argmax it is the first index
          of the largest value. "Right" for a sum or product means within float32 rounding.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Operator
          <span className="seg" role="group" aria-label="Operator" style={WRAP}>
            {OPS.map((o) => (
              <button key={o} aria-pressed={op === o} onClick={() => setOp(o)}>
                {FOLD_OPS[o].label}
              </button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          Start the accumulator at
          <span className="seg" role="group" aria-label="Start value" style={WRAP}>
            {STARTS.map((s) => (
              <button key={s} aria-pressed={start === s} onClick={() => setStart(s)}>
                {s === 'identity' ? `the identity, ${identity}` : START_LABEL[s]}
              </button>
            ))}
          </span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ flex: '1 1 18rem' }}>
          The list (up to 16 numbers)
          <input
            type="text"
            value={text}
            aria-label="The list"
            aria-invalid={bad}
            style={{ width: '100%', fontVariantNumeric: 'tabular-nums', borderColor: bad ? 'var(--fail)' : undefined }}
            onChange={(e) => {
              setText(e.target.value);
              const v = parseList(e.target.value);
              if (v) setValues(v);
            }}
          />
        </label>
        <span className="seg" role="group" aria-label="Lists" style={WRAP}>
          {PRESETS.map((p) => (
            <button key={p.label} onClick={() => pick(p.values)}>
              {p.label}
            </button>
          ))}
          <button
            onClick={() => {
              setSeed(seed + 1);
              pick(randomRow(8, seed));
            }}
          >
            a random row
          </button>
        </span>
      </div>
      {bad ? (
        <p className="t-fail" style={{ marginTop: 0 }}>
          Type between 1 and 16 numbers, separated by commas. The explorer keeps using the last list that parsed.
        </p>
      ) : null}

      <div className="cols" style={{ alignItems: 'start' }}>
        <div className="panel" style={{ overflowX: 'auto' }}>
          <div className="row" style={{ marginBottom: '0.6rem' }}>
            <button onClick={() => { setPlaying(false); setShown(0); }} disabled={shown === 0}>
              Restart
            </button>
            <button onClick={() => { setPlaying(false); setShown(Math.min(n, shown + 1)); }} disabled={done}>
              Step
            </button>
            <button className="primary" onClick={() => { if (done) setShown(0); setPlaying(!playing); }}>
              {playing ? 'Pause' : 'Play'}
            </button>
          </div>
          <table className="data" style={{ margin: 0 }} data-part="trace">
            <thead>
              <tr>
                <th>Iteration</th>
                <th className="num">input[i]</th>
                <th className="num">{isArg ? '(value, index)' : 'acc'} after it</th>
              </tr>
            </thead>
            <tbody>
              <tr data-row="start">
                <td>start</td>
                <td className="num muted">{trace.from === 1 ? 'element 0' : ''}</td>
                <td className="num">
                  <span className="chip reg">{accText(trace.init.value, trace.init.index)}</span>
                </td>
              </tr>
              {trace.steps.slice(0, shown).map((s) => (
                <tr key={s.i} data-row={s.i}>
                  <td>i = {s.i}</td>
                  <td className="num">{f32short(s.x)}</td>
                  <td className="num">
                    <span className="chip reg">{accText(s.acc, s.index)}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.5rem 0 0' }} data-readout="progress">
            {trace.from === 1
              ? `Started from element 0, so ${values.length} elements take ${n} iterations.`
              : `${values.length} elements, ${n} iterations.`}{' '}
            {done ? '' : `Shown: ${shown} of ${n}.`}
          </p>
        </div>
        <div className="panel" aria-live="polite" data-readout="result" data-ok={done ? (trace.ok ? '1' : '0') : ''}>
          {done ? (
            <>
              <p style={{ marginTop: 0 }}>
                <b>The loop's answer:</b>{' '}
                <span data-value="result">{accText(trace.result, trace.resultIndex)}</span>
              </p>
              <p>
                <b>The right answer:</b>{' '}
                <span data-value="truth">
                  {isArg ? `index ${trace.truthIndex} (value ${f64short(trace.truth)})` : f64short(trace.truth)}
                </span>
              </p>
              <p>
                {trace.ok ? <span className="chip ok">right</span> : <span className="chip fail">wrong</span>}{' '}
                {trace.ok
                  ? trace.start === 'identity' || trace.start === 'first'
                    ? 'As it must be with this start.'
                    : 'On this list. Try the other lists.'
                  : explainWrong(op, start)}
              </p>
            </>
          ) : (
            <p style={{ margin: 0 }} className="muted">
              The loop is running. Its answer appears after the last iteration.
            </p>
          )}
          <p style={{ marginBottom: 0 }} data-readout="random">
            The same loop and start on <b>{random.rows}</b> random rows of 8 logits: wrong on <b>{random.wrong}</b>.
            {random.wrong > 0 && random.wrong < 50 ? ' A test on random data could easily miss it.' : ''}
          </p>
        </div>
      </div>
    </Bench>
  );
}

function explainWrong(op: FoldOpId, start: StartId): string {
  if (start === 'zero' && (op === 'max' || op === 'argmax')) return 'Every value is below 0, so the start value wins and an answer that is not in the list comes out.';
  if (start === 'zero' && op === 'min') return 'Every value is above 0, so the start value wins.';
  if (start === 'zero' && op === 'product') return 'Anything times 0 is 0.';
  if (start === 'negmax' && op === 'min') return '−FLT_MAX is below every value, so it wins every comparison.';
  if (start === 'negmax' && op === 'sum') return 'The start value is part of the sum, and it is enormous.';
  if (start === 'negmax' && op === 'product') return 'The start value multiplies everything.';
  return 'The start value is not neutral for this operator.';
}

// ---------------------------------------------------------------- questions
export const questionsS101: Record<string, Question> = {
  'ch10/10.1/q-identity-max': {
    kind: 'choice',
    id: 'ch10/10.1/q-identity-max',
    prompt: (
      <>
        A max-logit kernel starts each thread's accumulator at <code>0.0f</code> and then takes the max with its
        elements. On which row of logits is its answer wrong?
      </>
    ),
    options: [
      { label: 'Any row that contains a positive logit.', why: 'Then some value beats 0 and the start never shows.' },
      { label: 'A row whose logits are all negative.', correct: true },
      { label: 'A row that contains one NaN.', why: 'That is a separate problem (section 10.2). The start value is wrong on a row with no NaN at all.' },
      { label: 'None: 0 is the identity of max.', why: '0 is the identity of addition. max(0, x) is 0, not x, whenever x is negative.' },
    ],
    explain: (
      <p>
        On an all-negative row the kernel returns 0, a value that is not in the row. The start of a max has to be at or
        below every possible input: −∞ in the book, or −FLT_MAX for finite floats. Rows with at least one positive
        logit hide the bug, and most rows have one, which is why a test on random logits can pass.
      </p>
    ),
    source: 'ch. 10 §10.1, p. 222',
  },
  'ch10/10.1/q-finite-identity': {
    kind: 'choice',
    id: 'ch10/10.1/q-finite-identity',
    prompt: (
      <>
        On a backend where infinities are undefined (Metal under fast math), which start value is safe for a max over
        finite floats, in every thread of a kernel?
      </>
    ),
    options: [
      { label: '0.0f', why: 'Wrong on any row that is all negative.' },
      { label: '-INFINITY', why: 'Exactly the value the compiler may assume never occurs in fast mode.' },
      { label: '-FLT_MAX, the most negative finite float', correct: true },
      { label: 'The thread\'s first element', why: 'Fine for one loop over a non-empty list. A thread or lane that has no element has nothing to start from.' },
    ],
    explain: (
      <p>
        Every finite float is at or above −FLT_MAX, so max(−FLT_MAX, x) = x for all of them, and nothing about it relies
        on infinities. Starting from the first element (footnote 1, p. 222) works in a sequential loop, but in a kernel
        the threads past the end of the input, and the padded lanes of a warp, have no first element. They still have
        to contribute something neutral.
      </p>
    ),
    source: 'ch. 10 §10.1, p. 222; Metal: notes/today/ch10-facts.md C3',
  },
  'ch10/10.1/q-reduce-order': {
    kind: 'choice',
    id: 'ch10/10.1/q-reduce-order',
    prompt: (
      <>
        <code>std::accumulate</code> and <code>std::reduce</code> sum the same <code>std::vector&lt;float&gt;</code>.
        Which one is allowed to give a different answer from run to run, and why?
      </>
    ),
    options: [
      { label: 'std::accumulate, because it uses a hidden accumulator type.', why: 'It folds left to right in one fixed order: the loop of Fig. 10.2.' },
      {
        label: 'std::reduce, because it may regroup and reorder the elements, and float addition is not associative.',
        correct: true,
      },
      { label: 'Neither: both add the same numbers.', why: 'The same numbers in a different grouping round differently.' },
      { label: 'Both, because float addition is not deterministic.', why: 'Each addition is deterministic. Only the grouping can change.' },
    ],
    explain: (
      <p>
        <code>std::accumulate</code> is the sequential loop. <code>std::reduce</code> may group and rearrange the
        elements in any order, and the C++ library documents its result as non-deterministic for floating-point
        addition. Those are the two properties section 10.2 asks of an operator before a parallel tree may compute it.
      </p>
    ),
    source: 'ch. 10 §10.1, p. 223 and §10.2, p. 224; cppreference: notes/today/ch10-facts.md A9',
  },
};

// ---------------------------------------------------------------- the section
export function S101() {
  const fold = useLabFile('ch10/snippets/fold.h');
  return (
    <>
      <p>
        A <b>reduction</b> turns a list of values into one value by applying a binary operator: an operation that takes
        two inputs and returns one. Summing is the book's example: the list 7.0, 2.1, 5.3, 9.0, 11.2 reduces to 34.6.
        The values could as well be integers, half-precision floats or characters. The structure of the computation is
        the same for all of them, and so is everything this chapter does to make it parallel.
      </p>
      <Source section="10.1" pages="221–222" />

      <h2>One loop for every reduction</h2>
      <p>
        Sequentially, a sum is the loop you would write without thinking: start a variable at 0, add each element to it
        in turn. Its value after each iteration is 7.0, 9.1, 14.4, 23.4 and finally 34.6. The book then pulls the
        operator out of the loop. Start an accumulator at a value called <code>IDENTITY</code>, and in each iteration
        replace it by <code>Operator(acc, input[i])</code>. Sum, product, max and min are all this one loop, with a
        different operator and a different starting value. N elements take N iterations.
      </p>
      <Source section="10.1" pages="222–223" figs="Figs. 10.1 and 10.2" />
      <Code
        tabs={[{ label: 'C++', code: fold ?? '// loading labs/ch10/snippets/fold.h' }]}
        highlight={[8, 9, 10]}
        caption={
          <>
            Fig. 10.2 written as a C++ function, plus the two reductions greedy decoding needs: the largest logit and
            the argmax (labs/ch10/snippets/fold.h). The unit tests compile and run it on the CPU and compare every float
            with the explorer below, bit for bit.
          </>
        }
      />

      <h2>Where the accumulator starts</h2>
      <p>
        The starting value is the operator's <b>identity</b>: the value that, combined with any input, gives that input
        back. For addition it is 0, for multiplication 1. For min it is +∞ and for max −∞: every value is at most +∞
        and at least −∞, so min(+∞, x) = x and max(−∞, x) = x. An identity is a convenience, not a requirement. An
        operator without one can start the accumulator at element 0 and begin the loop at element 1.
      </p>
      <NoteGroup>
        <Source section="10.1" pages="222" />
        <Aside>
          <p>
            Two slips on p. 222: the identity of the "minimum (minimum)" operator should read "minimum (maximum)", and
            "after iteration 5" means the fifth iteration, numbered 4 when counting from 0. Both are listed on the{' '}
            <a href={href('ch10', 'wrap')}>summary page</a>.
          </p>
        </Aside>
      </NoteGroup>
      <p>
        In a loop of your own, a wrong starting value is easy to spot. In a kernel it hides: the value 0 instead of
        −∞ gives the right max on almost every row you are likely to test with. Find the rows where it does not.
      </p>
      <Simplification>
        <p>
          One thread, no GPU: float32 is imitated by rounding to float32 after every operation. Max and min treat NaN as
          missing, like CUDA's <code>fmaxf</code>; the list cannot hold a NaN.
        </p>
      </Simplification>
      <FoldExplorer />
      <p>
        Three things should have come out of the explorer. Any start at or below every input works for max, which is why
        −FLT_MAX, the most negative finite float, is as good as −∞ on real logits. A start of 0 fails only on rows that
        are entirely negative, which random test data almost never produces. And starting from element 0 always works
        in a loop, but it assumes there is an element 0.
      </p>
      <Ask q={questionsS101['ch10/10.1/q-identity-max']} />

      <h2>What the identity becomes in a kernel</h2>
      <p>
        A softmax over a row of logits is two reductions in a row. First a max, with identity −∞; then a sum of exp(x
        − max), with identity 0; then one division per element. Argmax is a max over (value, index) pairs. In a GPU
        kernel the identity stops being a convenience. Thousands of threads each start an accumulator, and the threads
        whose index lies past the end of the row, or the lanes of a warp that have no element, must still hand
        something to the reduction. That something has to be neutral: the identity. Sections{' '}
        <a href={href('ch10', '10.7')}>10.7</a> and <a href={href('ch10', '10.9')}>10.9</a> depend on it.
      </p>
      <Beyond>
        <p>
          Softmax, argmax and the role of the identity for idle threads are ours; the book's chapter uses only sums.
        </p>
      </Beyond>
      <p>
        On Metal, the obvious identity for max is a trap. Under Metal's fast math the compiler may assume no value is
        infinite, so the behaviour of −∞ is undefined, and Apple documents the default math mode three different ways.
        This course's Metal kernels therefore start a max at −FLT_MAX. CUDA's fast-math option, by contrast, lists no
        such assumption.
      </p>
      <Today checked="2026-11-20">
        <p>
          MSL 4.1, §8.1: "If fast math is enabled the behavior of handling NaN or INF (as inputs or outputs) is
          undefined."
        </p>
        <p className="muted">ch10-facts C3, A11; ch05-facts C7</p>
      </Today>
      <Ask q={questionsS101['ch10/10.1/q-finite-identity']} />

      <h2>Reductions you do not write yourself</h2>
      <p>
        Reduction is common enough that every library has it. C++ has <code>std::accumulate</code> and{' '}
        <code>std::reduce</code>, two variants with different promises. On NVIDIA GPUs, the Thrust and CUB libraries
        offer heavily optimised reductions. The book writes its own anyway, because the road to a fast one passes
        through nearly every technique in the first part of the book.
      </p>
      <NoteGroup>
        <Source section="10.1" pages="223" />
        <Today checked="2026-11-20 (ch10-facts A9, A12)">
          <p>
            cppreference: for float addition, <code>std::reduce</code>'s "result is non-deterministic". Thrust and CUB
            now live under <code>nvidia.github.io/cccl/unstable/</code>.
          </p>
        </Today>
      </NoteGroup>
      <p>
        On AMD the counterparts are hipCUB and rocPRIM, with CUB's names in a different namespace and header, so a
        reduction written against CUB does not compile unchanged for HIP. On Metal, the documented building blocks are{' '}
        <code>simd_sum</code> and <code>simd_max</code>, which reduce across one SIMD-group; at the level of whole
        tensors, Metal Performance Shaders Graph offers reductions such as <code>reductionArgMaximum</code>.
      </p>
      <Today checked="2026-11-20 (infra-facts B11; ch10-facts C1, C5)">
        <p>
          hipCUB: "a thin, header-only wrapper library for rocPRIM". MPSGraph's argmax does not document its tie-break.
        </p>
      </Today>
      <Ask q={questionsS101['ch10/10.1/q-reduce-order']} />
    </>
  );
}
