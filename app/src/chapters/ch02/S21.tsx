import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source } from '../../components/notes';
import { href } from '../../lib/router';
import {
  BINS,
  BIN_LABEL,
  BIN_LONG,
  DECODE_ITEMS,
  OPS,
  OP_BODY,
  OP_LABEL,
  OP_NOTE,
  binCounts,
  classify,
  classifyDeps,
  combineReads,
  counts,
  defaults,
  reads,
  shape,
  stageItems,
  timesRead,
  underOrders,
  type Bin,
  type DecodeItem,
  type Op,
  type Params,
} from '../../sim/depend';
import { OrderScene } from './OrderScene';

const num = (v: number) => v.toLocaleString('en-US');
const plural = (n: number, one: string, many = one + 's') => `${num(n)} ${n === 1 ? one : many}`;

// ---------------------------------------------------------------- interactive A: what does one output read?
type Pick = { row: 'out' | 'combine'; i: number };

/** "x[4] and x[5]", "in[6] to in[10]", "I[9], I[10] and I[11]". */
function listCells(array: string, idx: number[]): string {
  const name = (i: number) => `${array}[${i}]`;
  if (idx.length === 0) return 'nothing';
  if (idx.length === 1) return name(idx[0]);
  const contiguous = idx.every((v, i) => i === 0 || v === idx[i - 1] + 1);
  if (contiguous && idx.length > 3) return `${name(idx[0])} to ${name(idx[idx.length - 1])}`;
  return idx.slice(0, -1).map(name).join(', ') + ' and ' + name(idx[idx.length - 1]);
}

const outName = (op: Op, k: number) => (op === 'mean' ? 'The mean' : `${shape(op).output.name}[${k}]`);
const combineName = (op: Op, c: number) => (op === 'mean' ? `partial sum ${c}` : `the sum of row ${c}`);

/** One sentence, built from the model, saying what the picked cell reads. */
function describe(op: Op, pick: Pick, p: Params): { text: string; bad: boolean } {
  if (pick.row === 'combine') {
    const r = combineReads(op, pick.i, p);
    const what = op === 'mean' ? `Partial sum ${pick.i}` : `The sum of row ${pick.i}`;
    return {
      text: `${what} is built from ${listCells(r.array, r.indices)}: ${plural(r.indices.length, 'input')}, and no other ${op === 'mean' ? 'partial sum' : 'row sum'}. Each of these can be computed without the others.`,
      bad: false,
    };
  }
  const r = reads(op, pick.i, p);
  const c = counts(op, pick.i, p);
  const own = r.inputs.map((a) => listCells(a.array, a.indices)).join(', and ');
  const name = outName(op, pick.i);
  if (r.chain.length) {
    const first = r.outputs[0];
    const o = shape(op, p).output.name;
    return {
      text:
        `${name} reads ${own} as it was before the loop, and it needs the new value of ${o}[${first}], which is another output. ` +
        (r.chain.length > 1
          ? `That one needs ${o}[${first - 1}], and so on back to ${o}[0]: ${plural(r.chain.length, 'output')} must exist first, in order.`
          : `1 output must exist first.`),
      bad: true,
    };
  }
  if (c.combines) {
    const parts = r.combines.map((x) => combineName(op, x));
    const which = parts.length === 1 ? parts[0] : `${plural(parts.length, 'partial sum')}`;
    return {
      text:
        (c.inputs ? `${name} reads ${own} itself and needs ` : `${name} reads no input itself. It needs `) +
        `${which} first, built from ${plural(c.viaCombines, 'input')} in all. It needs no other output.`,
      bad: false,
    };
  }
  return {
    text: `${name} reads ${own}: ${plural(c.inputs, 'input')}. It needs no other output, so its thread can run at any time.`,
    bad: false,
  };
}

/**
 * At phone widths the drawing is wider than its panel and scrolls inside it.
 * Keep x (in viewBox units) in view, so the picked cell is never off to the side.
 */
function useKeepInView(x: number, viewBoxWidth: number) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    const svg = el?.querySelector('svg');
    if (!el || !svg || el.scrollWidth <= el.clientWidth + 1) return;
    const px = (x / viewBoxWidth) * svg.getBoundingClientRect().width;
    el.scrollLeft = Math.max(0, px - el.clientWidth / 2);
  }, [x, viewBoxWidth]);
  return ref;
}

const WRAP = { whiteSpace: 'normal' } as const;
const VBW = 760;
const LEFT = 104;
const CH = 30;

function ReadExplorer() {
  const [op, setOp] = useState<Op>('grayscale');
  const [radius, setRadius] = useState(1);
  const [chunk, setChunk] = useState(16);
  const [picked, setPicked] = useState<Pick>({ row: 'out', i: 3 });
  const p: Params = useMemo(() => ({ ...defaults(op), radius, chunk }), [op, radius, chunk]);
  const s = shape(op, p);
  const pick: Pick =
    picked.row === 'combine' && s.combine
      ? { row: 'combine', i: Math.min(picked.i, s.combine.length - 1) }
      : { row: 'out', i: Math.min(picked.i, s.output.length - 1) };

  const r = pick.row === 'out' ? reads(op, pick.i, p) : null;
  const c = pick.row === 'out' ? counts(op, pick.i, p) : null;
  const direct = new Map<string, Set<number>>();
  const via = new Set<number>(); // inputs of the first input array reached through a combined value
  const needCombine = new Set<number>();
  const chain = new Set<number>(r?.chain ?? []);
  if (r) {
    r.inputs.forEach((a) => direct.set(a.array, new Set(a.indices)));
    r.combines.forEach((x) => {
      needCombine.add(x);
      combineReads(op, x, p).indices.forEach((i) => via.add(i));
    });
  } else {
    const cr = combineReads(op, pick.i, p);
    direct.set(cr.array, new Set(cr.indices));
  }
  const said = describe(op, pick, p);
  const orders = useMemo(() => underOrders(op, undefined, p), [op, p]);
  const allMatch = orders.every((o) => o.matches);
  const bin = classify(op);

  // Geometry: one cell width for the longest input array; outputs and combined values span the inputs they sit under.
  const maxLen = Math.max(...s.inputs.map((a) => a.length));
  const cw = Math.min(40, (VBW - LEFT - 8) / maxLen);
  const span = (len: number) => (maxLen % len === 0 ? maxLen / len : 1);
  const cellX = (len: number, i: number, aligned: boolean) => LEFT + i * cw * (aligned ? span(len) : 1);
  const cellW = (len: number, aligned: boolean) => cw * (aligned ? span(len) : 1) - 3;
  const inY = (a: number) => 22 + a * 62;
  const combineY = inY(s.inputs.length) + 6;
  const outY = (s.combine ? combineY + 62 : inY(s.inputs.length)) + 6;
  const height = outY + CH + 34;
  const mid = (len: number, i: number, aligned: boolean) => cellX(len, i, aligned) + cellW(len, aligned) / 2;

  const cell = (
    key: string,
    x: number,
    y: number,
    w: number,
    label: string,
    state: 'idle' | 'read' | 'via' | 'needed' | 'chain' | 'picked',
    onPick?: () => void,
    aria?: string,
    sub?: string,
  ) => {
    const look = {
      idle: { fill: 'var(--cell)', stroke: 'var(--rule)', ink: 'var(--ink-2)', sw: 1 },
      read: { fill: 'var(--global)', stroke: 'var(--global)', ink: 'var(--page)', sw: 1 },
      via: { fill: 'var(--global-wash)', stroke: 'var(--global)', ink: 'var(--ink)', sw: 1 },
      needed: { fill: 'var(--page)', stroke: 'var(--ink)', ink: 'var(--ink)', sw: 2.2 },
      chain: { fill: 'var(--fail-wash)', stroke: 'var(--fail)', ink: 'var(--fail)', sw: 1.6 },
      picked: { fill: 'var(--ink)', stroke: 'var(--ink)', ink: 'var(--page)', sw: 1 },
    }[state];
    return (
      <g
        key={key}
        data-cell={key}
        data-state={state}
        role={onPick ? 'button' : undefined}
        tabIndex={onPick ? 0 : undefined}
        aria-label={aria}
        aria-pressed={onPick ? state === 'picked' : undefined}
        style={onPick ? { cursor: 'pointer' } : undefined}
        onClick={onPick}
        onKeyDown={
          onPick
            ? (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onPick();
                }
              }
            : undefined
        }
      >
        <rect x={x} y={y} width={w} height={CH} rx="2" fill={look.fill} stroke={look.stroke} strokeWidth={look.sw} />
        <text x={x + w / 2} y={y + CH / 2 + 4} fontSize="11.5" textAnchor="middle" fontWeight={state === 'idle' ? 400 : 650}
          style={{ fill: look.ink, fontVariantNumeric: 'tabular-nums', pointerEvents: 'none' }}>
          {label}
        </text>
        {sub != null ? (
          <text x={x + w / 2} y={y + CH + 11} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-3)', pointerEvents: 'none' }}>
            {sub}
          </text>
        ) : null}
      </g>
    );
  };

  const rowLabel = (y: number, main: string, sub?: string) => (
    <g>
      <text x={LEFT - 10} y={y + (sub ? 13 : 19)} fontSize="12" fontWeight="600" textAnchor="end">{main}</text>
      {sub ? (
        <text x={LEFT - 10} y={y + 26} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{sub}</text>
      ) : null}
    </g>
  );

  const lines: ReactNode[] = [];
  const outLen = s.output.length;
  if (pick.row === 'out' && r) {
    const fromX = mid(outLen, pick.i, true);
    // Without a combine row the inputs carry a count under each cell; stop the lines below it.
    const below = s.combine ? 0 : 15;
    r.inputs.forEach((a) => {
      const ai = s.inputs.findIndex((x) => x.name === a.array);
      a.indices.forEach((i) =>
        lines.push(
          <line key={`l-${a.array}-${i}`} x1={fromX} y1={outY} x2={mid(s.inputs[ai].length, i, false)} y2={inY(ai) + CH + below}
            stroke="var(--global)" strokeWidth="1.3" opacity="0.75" />,
        ),
      );
    });
    if (s.combine)
      r.combines.forEach((x) =>
        lines.push(
          <line key={`lc-${x}`} x1={fromX} y1={outY} x2={mid(s.combine!.length, x, true)} y2={combineY + CH}
            stroke="var(--ink)" strokeWidth="1.5" />,
        ),
      );
    // The chain: one hop from each needed output to the one before it, back to the start.
    [pick.i, ...r.chain].slice(0, -1).forEach((k) =>
      lines.push(
        <path key={`ch-${k}`} d={`M ${mid(outLen, k, true) - 4} ${outY + CH + 2} q ${-cw / 2 + 4} 13 ${-cw + 8} 0`}
          fill="none" stroke="var(--fail)" strokeWidth="1.6" markerEnd="url(#dep-arrow)" />,
      ),
    );
  }
  // A bracket from each lit combined value up to the inputs it is built from.
  if (s.combine) {
    const lit = pick.row === 'combine' ? [pick.i] : [...needCombine];
    lit.forEach((x) => {
      const idx = combineReads(op, x, p).indices;
      const x0 = cellX(maxLen, idx[0], false);
      const x1 = cellX(maxLen, idx[idx.length - 1], false) + cellW(maxLen, false);
      const y = inY(0) + CH + 16;
      lines.push(
        <path key={`br-${x}`} d={`M ${x0} ${y - 4} V ${y} H ${x1} V ${y - 4} M ${mid(s.combine!.length, x, true)} ${y} V ${combineY}`}
          fill="none" stroke="var(--global)" strokeWidth="1.3" />,
      );
    });
  }

  const choose = (next: Op) => {
    setOp(next);
    setPicked({ row: 'out', i: Math.min(5, shape(next).output.length - 1) });
  };
  const scroller = useKeepInView(
    pick.row === 'combine' && s.combine ? mid(s.combine.length, pick.i, true) : mid(outLen, pick.i, true),
    VBW,
  );

  return (
    <Bench
      title="What does one output read?"
      task="Pick an operation, then click an output cell. Before you click, say how many inputs will light up and whether anything will turn red."
      foot="Lit cells are input elements the output's own thread reads. A red cell is another output that must exist first. The small numbers under the inputs count how many outputs read each one."
    >
      <div data-explorer data-op={op}>
        <div className="row" role="group" aria-label="Operation" style={{ gap: '0.4rem', marginBottom: '0.8rem' }}>
          {OPS.map((o) => (
            <button key={o} className="quiet" data-op-choice={o} aria-pressed={o === op} onClick={() => choose(o)}>
              {OP_LABEL[o]}
            </button>
          ))}
        </div>
        <div className="row" style={{ marginBottom: '0.6rem', minHeight: '2.4rem' }}>
          <span>
            {s.output.length > 1 ? <span className="muted">for each k: </span> : null}
            <code data-out="body">{OP_BODY[op]}</code>
            {OP_NOTE[op] ? <span className="muted"> ({OP_NOTE[op]})</span> : null}
          </span>
          {op === 'blur' ? (
            <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}>
              <span>
                Radius r = <b data-out="radius">{radius}</b>
              </span>
              <input type="range" min="0" max="4" step="1" value={radius} aria-label="Blur radius"
                onChange={(e) => setRadius(Number(e.target.value))} />
            </label>
          ) : null}
          {op === 'mean' ? (
            <span className="row" style={{ gap: '0.5rem' }}>
              <span className="muted">Split the sum into partial sums of</span>
              <span className="seg" role="group" aria-label="Elements per partial sum">
                {[16, 8, 4, 2].map((v) => (
                  <button key={v} data-chunk={v} aria-pressed={chunk === v} onClick={() => setChunk(v)}>
                    {v}
                  </button>
                ))}
              </span>
            </span>
          ) : null}
        </div>

        <div className="panel" style={{ overflowX: 'auto' }} ref={scroller}>
          <svg viewBox={`0 0 ${VBW} ${height}`} role="group" aria-label={`Cells of ${OP_LABEL[op]}: inputs above, outputs below`}
            style={{ width: '100%', minWidth: 620, maxWidth: VBW, display: 'block' }}>
            <defs>
              <marker id="dep-arrow" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 8 4 L 0 8 z" fill="var(--fail)" />
              </marker>
            </defs>
            {lines}
            {s.inputs.map((a, ai) => (
              <g key={a.name}>
                {rowLabel(
                  inY(ai),
                  s.inPlace ? `${a.name} before` : a.name,
                  op === 'grayscale' ? 'input, r g b' : s.inPlace ? 'the values read' : 'input',
                )}
                {Array.from({ length: a.length }, (_, i) => {
                  const isRead = direct.get(a.name)?.has(i);
                  const isVia = ai === 0 && via.has(i);
                  return cell(
                    `in:${a.name}:${i}`,
                    cellX(a.length, i, false),
                    inY(ai),
                    cellW(a.length, false),
                    op === 'grayscale' ? 'rgb'[i % 3] : String(i),
                    isRead ? 'read' : isVia ? 'via' : 'idle',
                    undefined,
                    undefined,
                    s.combine ? undefined : String(timesRead(op, a.name, i, p)),
                  );
                })}
              </g>
            ))}
            {s.combine ? (
              <g>
                {rowLabel(combineY, s.combine.name, 'combined first')}
                {Array.from({ length: s.combine.length }, (_, x) =>
                  cell(
                    `combine:${x}`,
                    cellX(s.combine!.length, x, true),
                    combineY,
                    cellW(s.combine!.length, true),
                    op === 'mean' ? `sum ${x}` : `sum of row ${x}`,
                    pick.row === 'combine' && pick.i === x ? 'picked' : needCombine.has(x) ? 'needed' : 'idle',
                    () => setPicked({ row: 'combine', i: x }),
                    `${combineName(op, x)}: show what it reads`,
                  ),
                )}
              </g>
            ) : null}
            <g>
              {rowLabel(outY, s.inPlace ? `${s.output.name} after` : s.output.name, s.inPlace ? 'same memory' : 'output')}
              {Array.from({ length: outLen }, (_, k) =>
                cell(
                  `out:${k}`,
                  cellX(outLen, k, true),
                  outY,
                  cellW(outLen, true),
                  op === 'mean' ? 'mean' : String(k),
                  pick.row === 'out' && pick.i === k ? 'picked' : chain.has(k) ? 'chain' : 'idle',
                  () => setPicked({ row: 'out', i: k }),
                  `Output ${k}: show what it reads`,
                ),
              )}
            </g>
          </svg>
        </div>

        <div className="cols" style={{ marginTop: '0.8rem' }}>
          <div className={'narrate' + (said.bad ? ' bad' : '')} data-out="said" role="status">
            {said.text}
          </div>
          <div className="panel">
            <table className="threads">
              <tbody>
                <tr>
                  <td style={WRAP}>Inputs it reads itself</td>
                  <td>
                    <output data-out="inputs">{c ? c.inputs : combineReads(op, pick.i, p).indices.length}</output>
                  </td>
                </tr>
                {s.combine ? (
                  <tr>
                    <td style={WRAP}>Combined values needed first</td>
                    <td>
                      <output data-out="combines">{c ? c.combines : 0}</output>
                    </td>
                  </tr>
                ) : null}
                <tr>
                  <td style={WRAP}>Outputs needed first</td>
                  <td>
                    <output data-out="outputs-first" className={c && c.outputsFirst ? 't-fail' : undefined}>
                      {c ? c.outputsFirst : 0}
                    </output>
                  </td>
                </tr>
                <tr>
                  <td style={WRAP}>Shape of the operation</td>
                  <td style={WRAP}>
                    <output data-out="bin" data-bin={bin}>{BIN_LABEL[bin]}</output>
                  </td>
                </tr>
                <tr>
                  <td style={WRAP}>One thread per output, {orders.length} thread orders</td>
                  <td>
                    <output data-out="orders" data-match={allMatch ? 'all' : 'some'}>
                      {allMatch ? (
                        <span className="chip ok">same result every time</span>
                      ) : (
                        <span className="chip fail">
                          {orders.filter((o) => !o.matches).length} of {orders.length} differ from the loop
                        </span>
                      )}
                    </output>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive B: sort a decode step
const SHORT: Record<Bin, string> = {
  elementwise: 'Per element',
  manyreads: 'Per output, many reads',
  combine: 'Combine first',
  chain: 'Chain',
};

const course = (id: string, text: string) => <a href={href(id)}>{text}</a>;

/** What the reader sees once an item is in the right bin: its read set from the model, and where its kernel is written. */
const PLACED: Record<string, (it: DecodeItem) => ReactNode> = {
  residual: (it) => (
    <>
      One output reads {it.deps.reads(4096)} inputs, <code>x[i]</code> and <code>r[i]</code>, whatever the size of the
      tensor. This is the vector addition of this chapter under another name, and one of the three operations of
      milestone M0 (<a href={href('ch02', 'labs')}>lab 2.3</a>).
    </>
  ),
  bias: (it) => (
    <>
      One output reads {it.deps.reads(4096)} inputs: its own element and one bias value. Every row reads the same bias
      values, and shared reads are harmless. Milestone M0 (<a href={href('ch02', 'labs')}>lab 2.3</a>).
    </>
  ),
  scale: (it) => (
    <>
      One output reads {it.deps.reads(4096)} input. Milestone M0. The kernel in the book's exercise 9,{' '}
      <code>b[i] = 2.7f*a[i] - 4.3f</code>, is this operation with an offset added.
    </>
  ),
  gelu: (it) => (
    <>
      One output reads {it.deps.reads(4096)} input. The arithmetic for one element is longer than an addition, and the
      kernel has the same shape as this chapter's: an index, a guard, one element.
    </>
  ),
  linear: (it) => (
    <>
      Outputs do not need each other, but one output reads {num(it.deps.reads(4096))} values when d = 4096: a whole
      row of <code>W</code> and all of <code>x</code>. Each thread now has a loop inside it.{' '}
      {course('ch03', 'Chapter 3')} writes this kernel, {course('ch05', 'chapter 5')} cuts its memory traffic and{' '}
      {course('ch15', 'chapter 15')} optimises it.
    </>
  ),
  layernorm: (it) => (
    <>
      Each output reads {it.deps.reads(4096)} values of its own (its element, a gain and a bias) and needs{' '}
      {it.deps.combines} values combined from the whole row first, the mean and the variance. Computing those is a
      reduction, the pattern of {course('ch10', 'chapter 10')}. After that the rest is per element.
    </>
  ),
  softmax: (it) => (
    <>
      Each output reads {it.deps.reads(50000)} logit of its own and needs {it.deps.combines} values combined from the
      whole vocabulary first: the largest logit m, subtracted before the exponential so that no exponential
      overflows, and then the sum of the exponentials. Each is a reduction: {course('ch10', 'chapter 10')},
      milestone M4. The explorer above uses the plain formula, which needs only the sum.
    </>
  ),
  argmax: (it) => (
    <>
      There is one output, and it is itself {it.deps.combines} value combined from the whole vocabulary. One thread
      per output would be one thread. The parallelism has to go inside the combine: {course('ch10', 'chapter 10')},
      milestone M4.
    </>
  ),
  cumprob: () => (
    <>
      <code>c[i]</code> needs <code>c[i-1]</code>: the in-place running sum of the explorer above.{' '}
      {course('ch11', 'Chapter 11')} replaces the chain with a different algorithm, and {course('x1', 'module X1')}{' '}
      uses it for top-p.
    </>
  ),
  kvappend: (it) => (
    <>
      A copy. Each element of the new key and value vectors goes to the next free row of the cache: {it.deps.reads(4096)}{' '}
      read per output, and no output needs another. Which row is free is a number the host passes in. What the cache
      is for is {course('ch20', 'chapter 20')}.
    </>
  ),
  genloop: () => (
    <>
      Token t is computed from tokens 0 to t − 1, so no kernel can run step t before step t − 1 has finished. This
      is the loop around the step, not an operation inside it: kernels make each step faster, and the steps stay in
      order. {course('ch20', 'Chapter 20')} covers what can be kept between steps and how to batch across sequences.
    </>
  ),
};

/** Why a wrong bin is wrong, said from the item's dependency facts. */
function whyNot(it: DecodeItem, pick: Bin): ReactNode {
  const d = it.deps;
  const actual = classifyDeps(d);
  if (pick === 'chain')
    return <>No output of this operation needs another output of the same operation. A chain is exactly that need.</>;
  if (actual === 'chain')
    return <>Look at the line of work: the result on the left needs an earlier result of the same operation.</>;
  if (pick === 'combine')
    return <>Nothing has to be combined across a whole row or array before an output can be computed. Each output gets its inputs directly.</>;
  if (actual === 'combine')
    return (
      <>
        An output cannot be computed from its own inputs alone. It needs {plural(d.combines, 'value')} built from the
        whole row first.
      </>
    );
  if (pick === 'manyreads')
    return (
      <>
        One output reads {plural(d.reads(4096), 'input')} however large the tensor is. The number does not grow with
        the {it.sizeName}.
      </>
    );
  return (
    <>
      Count what one output reads: {num(d.reads(4096))} values when the {it.sizeName} is 4,096, and twice as many when
      it doubles. That is a loop inside each thread, not one line.
    </>
  );
}

function SortDecodeStep() {
  const [picks, setPicks] = useState<Record<string, Bin | undefined>>({});
  const placed = DECODE_ITEMS.filter((it) => picks[it.id] === classifyDeps(it.deps)).length;
  const tally = binCounts();
  const forward = stageItems('forward').length;
  const sampling = stageItems('sampling').length;
  const inForward = binCounts('forward');
  return (
    <Bench
      title="Sort a decode step"
      task={`Eleven items from one decode step, in order: ${forward} operations of the forward pass, ${sampling} of sampling, and the loop around the step. Put each in the bin that describes what one of its outputs needs.`}
      foot="The list and the sorting are ours, not the book's. The numbers in the feedback come from the same dependency model as the explorer above."
    >
      <div data-sort>
        <div className="panel" style={{ marginBottom: '0.8rem' }}>
          <table className="threads" style={{ whiteSpace: 'normal' }}>
            <tbody>
              {BINS.map((b) => (
                <tr key={b}>
                  <td style={{ whiteSpace: 'nowrap', fontWeight: 600 }}>{SHORT[b]}</td>
                  <td style={{ whiteSpace: 'normal' }}>{BIN_LONG[b]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <div className="row" style={{ justifyContent: 'space-between', marginBottom: '0.2rem' }}>
            <b>Eleven items</b>
            <span className="muted" data-sort-score>
              {placed} of {DECODE_ITEMS.length} placed
            </span>
          </div>
          {DECODE_ITEMS.map((it, n) => {
            const pick = picks[it.id];
            const answer = classifyDeps(it.deps);
            const done = pick === answer;
            return (
              <div key={it.id} data-card={it.id} data-solved={done ? 'true' : 'false'}
                style={{ padding: '0.75rem 0', borderTop: n ? '1px solid var(--rule-soft)' : undefined }}>
                <p style={{ margin: '0 0 0.5rem', maxWidth: '46rem', fontSize: '0.92rem' }}>
                  <span className="muted">{n + 1}.</span> <b>{it.label}</b> <code>{it.body}</code>
                </p>
                <div className="row" style={{ gap: '0.4rem' }} role="group" aria-label={`Bin for ${it.label}`}>
                  {BINS.map((b) => (
                    <button key={b} className="quiet" data-choice={b} aria-pressed={pick === b} disabled={done && b !== answer}
                      onClick={() => (done ? undefined : setPicks((s) => ({ ...s, [it.id]: b })))}>
                      {SHORT[b]}
                    </button>
                  ))}
                </div>
                {pick ? (
                  <p data-feedback={it.id} role="status" style={{ margin: '0.5rem 0 0', maxWidth: '46rem' }}>
                    {done ? (
                      <>
                        <span className="chip ok">placed</span> {PLACED[it.id](it)}
                      </>
                    ) : (
                      <>
                        <span className="chip fail">not this one</span> {whyNot(it, pick)}
                      </>
                    )}
                  </p>
                ) : null}
              </div>
            );
          })}
        </div>
        <div className="row" style={{ marginTop: '0.8rem', justifyContent: 'flex-end' }}>
          <button className="quiet" onClick={() => setPicks({})}>
            Start again
          </button>
        </div>
        {placed === DECODE_ITEMS.length ? (
          <div className="panel" style={{ marginTop: '0.8rem' }} data-sort-done>
            <p style={{ margin: 0 }}>
              <b>All eleven placed.</b> Count the bins: {tally.elementwise} per element, {tally.manyreads} with many
              reads, {tally.combine} that combine first, {tally.chain} chains. Of the {forward} operations inside the
              forward pass, {inForward.elementwise} are per element: by number of operations, most of the forward pass
              is the kernel shape of this chapter. The one operation with many reads, the matmul, is where the time
              goes, and it takes three chapters. The combine bin holds the reductions: layer norm inside the forward
              pass, and the softmax and argmax of sampling. The chain bin holds the running sum for top-p and the
              generation loop, which is the loop around the step rather than an operation inside it.
            </p>
          </div>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
// Numbers the questions state, from the model.
const BLUR2: Params = { ...defaults('blur'), radius: 2 };
const BLUR2_READS = counts('blur', 8, BLUR2).inputs;
const BLUR2_TIMES = timesRead('blur', 'in', 8, BLUR2);

export const questionsS21: Record<string, Question> = {
  'ch02/2.1/q-criterion': {
    kind: 'choice',
    id: 'ch02/2.1/q-criterion',
    prompt: <>Which condition makes it safe to give every iteration of a loop its own thread?</>,
    options: [
      {
        label: 'The iterations read disjoint inputs.',
        why: 'Sufficient for the reads, and far too strong. Blur reads overlap heavily and it is fine, as the explorer shows.',
      },
      {
        label: 'No iteration reads a location that another iteration writes.',
        why: 'Only half of it. The loop for i: m = x[i] meets it, since nothing it reads is written, yet every iteration writes m, and the value left there depends on which thread ran last.',
      },
      {
        label: 'Each iteration writes only locations that no other iteration reads or writes.',
        correct: true,
        why: 'Then nothing a thread reads can change under it, and nothing it writes can be overwritten, so the result cannot depend on when any thread ran.',
      },
      {
        label: 'Every iteration does the same amount of work.',
        why: 'Uneven work makes threads finish at different times. It does not change what any of them computes.',
      },
      {
        label: 'The loop body has no branches.',
        why: 'A branch is private to the thread that takes it. Blur tests for the edge of the array in every thread.',
      },
    ],
    explain: (
      <p>
        In the book's picture every output has its own arrows from the input and no arrow runs from one output to
        another. Reads may overlap as much as they like, because a read changes nothing. The condition is about
        writes, and it has two halves: each iteration writes only its own output, and nothing it reads is written by
        another iteration. Drop the first half and two threads can write the same location, so the last one to run
        decides what stays there.
      </p>
    ),
    source: '§2.1, pp. 21–23 (Fig. 2.2); the wording of the test is ours',
  },
  'ch02/2.1/q-inplace': {
    kind: 'choice',
    id: 'ch02/2.1/q-inplace',
    prompt: (
      <>
        Loop 1 is <code>for i: y[i] = x[i-1] + x[i]</code>. Loop 2 is <code>for i: x[i] = x[i-1] + x[i]</code>. Which
        can be run with one thread for each <code>i</code> and be certain to give the sequential answer?
      </>
    ),
    options: [
      {
        label: 'Only loop 1.',
        correct: true,
      },
      {
        label: 'Both. They do the same arithmetic.',
        why: 'The same arithmetic on different values. In loop 2, x[i-1] is the value iteration i − 1 has just written.',
      },
      {
        label: 'Only loop 2, because it needs no second array.',
        why: 'Writing into the array it reads is exactly what breaks it.',
      },
      {
        label: 'Neither, because neighbouring iterations read the same element.',
        why: 'In loop 1 both read x[i-1], and nobody writes it. Shared reads are harmless.',
      },
    ],
    explain: (
      <p>
        Loop 2 reads what the previous iteration wrote, so its result depends on the order in which the threads run.
        Run in loop order it is a running total, also called a prefix sum. Run with the last thread first it gives
        loop 1's answer, as the animation shows. A prefix sum can be computed in parallel, but by a different
        algorithm, which is <a href={href('ch11')}>chapter 11</a>.
      </p>
    ),
    source: 'our example, by the method of §2.1 (pp. 21–23)',
  },
  'ch02/2.1/q-blur-reads': {
    kind: 'numeric',
    id: 'ch02/2.1/q-blur-reads',
    prompt: (
      <>
        A 1D blur with radius 2 computes each output from {BLUR2_READS} neighbouring inputs, with one thread per
        output. Across all threads, how many times is one input element in the interior of the array read?
      </>
    ),
    answer: BLUR2_TIMES,
    unit: 'times',
    explain: (
      <p>
        An input at position j lies in the neighbourhood of the outputs from j − 2 to j + 2, which is {BLUR2_TIMES}{' '}
        outputs, so {BLUR2_TIMES} threads read it. The blur is still fully data parallel: no thread needs another
        thread's result. But the same bytes are fetched from memory {BLUR2_TIMES} times. Removing that kind of
        repeated fetch is the subject of <a href={href('ch05', '5.3')}>chapter 5</a>.
      </p>
    ),
    source: '§2.1, p. 21 (blur as the neighbourhood case); the count is ours',
  },
  'ch02/2.1/q-average': {
    kind: 'choice',
    id: 'ch02/2.1/q-average',
    prompt: (
      <>
        The book says the average brightness of an image appears global but can be broken into smaller computations
        that run independently. What makes that possible?
      </>
    ),
    options: [
      {
        label: 'Additions can be regrouped: sum chunks of the image independently, then combine the partial sums.',
        correct: true,
      },
      {
        label: 'Each pixel can be given its own thread, which adds its value straight into one shared total.',
        why: 'Many threads reading and writing one location is a race: updates get lost. Chapter 9 is about exactly this.',
      },
      {
        label: 'The image can be made smaller first, so that fewer pixels need to be read.',
        why: 'That computes the average of a different image.',
      },
      {
        label: 'It is not possible exactly. The book is describing an approximation.',
        why: 'The decomposition is exact in real arithmetic. See the note on rounding in the explanation.',
      },
    ],
    explain: (
      <p>
        No partial sum needs another partial sum, so they are independent, and one combine step at the end gives the
        total. Try it in the explorer with partial sums of 4. One caution for later: floating-point addition is not
        exactly associative, so regrouping a sum can change its last bits. The book takes that up in{' '}
        <a href={href('appA')}>appendix A</a>, and <a href={href('ch10')}>chapter 10</a> builds the combine step
        properly.
      </p>
    ),
    source: '§2.1, p. 21; the rounding caution is from appendix A, pp. 593–594',
  },
};

export function S21() {
  return (
    <>
      <p>
        Before you write a kernel you need to be able to look at a computation and say whether it can be cut into
        pieces that do not need each other. The book calls that property data parallelism: computations on different
        parts of the data can be done independently of each other, and therefore at the same time. It matters because
        when an application is slow today the usual reason is the amount of data, and an application rich in data
        parallelism gets faster in proportion to the hardware it is given.
      </p>
      <Source section="2.1" pages="21" />

      <h2>Three cases, graded</h2>
      <p>
        The book grades the idea with three image operations. Converting a colour pixel to gray needs that pixel and
        nothing else. Blurring needs a small neighbourhood of pixels around each output. The average brightness of a
        whole image looks global, since one number depends on every pixel, but it can be broken into smaller
        computations that run independently.
      </p>
      <Source section="2.1" pages="21" />
      <p>
        Take the first case exactly. A pixel holds three fractions r, g and b, each between 0 for dark and 1 for full
        intensity. Its luminance is a weighted sum, <code>L = 0.299 r + 0.587 g + 0.114 b</code>. With the pixels in an
        input array <code>I</code> and the luminances in an output array <code>O</code>, the value{' '}
        <code>O[k]</code> depends on <code>I[k]</code> and on no other pixel. In memory a row of pixels is stored
        interleaved, r g b r g b and so on, so pixel k is three consecutive values.
      </p>
      <Source section="2.1" pages="21–23" figs="Eq. 2.1, Fig. 2.2" />
      <p>
        Here is the property as a test you can apply to a loop: <b>no output is computed from another output</b>.
        Inputs may be read by as many threads as you like, and the reads of two outputs may overlap. What must not
        happen is one thread needing a value that another thread of the same launch produces.
      </p>
      <Aside>
        <p>
          The wording of the test is ours. The book gives the definition and a picture in which no arrow runs between
          two outputs, and leaves the test implicit.
        </p>
      </Aside>

      <h2>What one output reads</h2>
      <p>
        The explorer applies the test to eight operations. Start with grayscale and vector addition, then drag the
        blur radius and watch which of the two counts moves. Leave the two sums and the softmax for last.
      </p>
      <Simplification>
        <p>
          The explorer shows which elements are read. It says nothing about how long a read takes, or whether a
          second read of the same element is served from a cache. The thread orders it tries run one whole thread
          after another; real threads interleave more finely.
        </p>
      </Simplification>
      <ReadExplorer />
      <p>
        Three things to take from it. Overlapping reads are fine: a blur of radius 4 reads nine inputs per output and
        "outputs needed first" stays at 0. The mean and the softmax need a value combined from many elements before
        any output can be finished, and the pieces of that combine are themselves independent. The in-place running
        sum is the only one with red in it: every output waits for the one before.
      </p>
      <Ask q={questionsS21['ch02/2.1/q-criterion']} />
      <Ask q={questionsS21['ch02/2.1/q-blur-reads']} label="Work it out" />
      <Ask q={questionsS21['ch02/2.1/q-average']} />

      <h2>Why another thread's result is the problem</h2>
      <p>
        When a kernel runs, you do not control the order in which its threads execute. An operation that reads only
        inputs gives the same answer in any order. An operation in which one thread reads what another writes does
        not. The animation runs two nearly identical loops in two orders each.
      </p>
      <Aside>
        <p>
          The example is ours. The 8 numbers are the ones the book uses when it introduces the running total in
          chapter 11 (p. 252).
        </p>
      </Aside>
      <OrderScene />
      <Ask q={questionsS21['ch02/2.1/q-inplace']} />

      <h2>One decode step, sorted by dependency</h2>
      <p>
        The same test sorts the operations you already know. Take one decode step of a transformer and ask, for each
        operation, what one of its outputs needs. Four shapes cover them. The first is exactly the kernel of this
        chapter, and it is what milestone M0 of the course project is made of. The other three name the chapters that
        deal with them.
      </p>
      <Beyond>
        <p>
          The book's examples in this section are image operations. The mapping to the operations of a language model
          is ours.
        </p>
      </Beyond>
      <SortDecodeStep />

      <h2>Data parallelism and task parallelism</h2>
      <p>
        The book separates data parallelism from task parallelism, which comes from splitting an application into
        different tasks that can run at the same time. Its example is an application that needs a vector addition and
        a matrix-vector multiplication: two tasks, and task parallelism if the two can run independently. Data
        transfers and I/O are tasks as well. Of the two kinds, the book says data parallelism is the main source of
        speed and of scalability. It grows with the data: twice the pixels is twice the independent work, while the
        number of tasks stays what it was. In inference terms (our example, not the book's), a batch is data
        parallelism, and tokenising the next request while the previous step runs on the GPU is task parallelism.
      </p>
      <Source section="2.1" pages="23" />

      <h2>Parallel thinking</h2>
      <p>
        Writing data-parallel code means organising the computation around the data, so that the independent pieces
        are visible and can be handed to threads. For grayscale the pieces are obvious. For the average they had to
        be found. The book calls the skill of finding them in harder cases parallel thinking, and most of what
        follows in the book is practice at it.
      </p>
      <Source section="2.1" pages="21, 23" />
      <p>
        Nothing in this section depends on the GPU you use. Whether one output needs another is a property of the
        computation, and it is the same question on CUDA, HIP and Metal. The grayscale and blur kernels themselves
        are written in <a href={href('ch03')}>chapter 3</a>. This chapter uses something simpler, the addition of two
        vectors, and before the kernel it has to set up the program around it:{' '}
        <a href={href('ch02', '2.2')}>section 2.2</a>.
      </p>
    </>
  );
}
