import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import {
  OP_BODY,
  OP_NOTE,
  defaults,
  reads,
  runSequential,
  runThreads,
  threadOrder,
  type OrderName,
  type Params,
} from '../../sim/depend';

// "Same loop, another order": two loops over 8 numbers, each run with one thread
// per iteration in loop order and then backwards. Every number drawn comes from
// sim/depend.ts. Six beats, one narration script each
// (content/ch02/narration/order-N.txt).
//
// The 8 numbers are the ones the book uses for prefix sum in chapter 11 (p. 252),
// so the in-place loop's answer here, 3 4 11 11 15 16 22 25, is the book's too.
export const SCENE_X = [3, 1, 7, 0, 4, 1, 6, 3];
const N = SCENE_X.length;
type Loop = 'neighbour' | 'inplace';
const params = (op: Loop): Params => ({ ...defaults(op), n: N });
const DATA = { x: SCENE_X };
export const SCENE_WANT: Record<Loop, number[]> = {
  neighbour: runSequential('neighbour', DATA, params('neighbour')),
  inplace: runSequential('inplace', DATA, params('inplace')),
};

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

const MONO = { fontVariationSettings: "'MONO' 1, 'CASL' 0" };
const C = 56; // cell width
const H = 34; // cell height
const X0 = 222; // left edge of the cells
const W = 720;

interface CellLook {
  text: string;
  fill?: string;
  stroke?: string;
  ink?: string;
  bold?: boolean;
  width?: number;
}

function Row({ y, label, sub, cells }: { y: number; label: string; sub?: string; cells: CellLook[] }) {
  return (
    <g>
      <text x={X0 - 12} y={y + (sub ? 15 : 21)} fontSize="12.5" fontWeight="600" textAnchor="end">{label}</text>
      {sub ? (
        <text x={X0 - 12} y={y + 29} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{sub}</text>
      ) : null}
      {cells.map((c, i) => (
        <g key={i}>
          <rect x={X0 + i * C + 1} y={y} width={C - 4} height={H} rx="2" fill={c.fill ?? 'var(--cell)'}
            stroke={c.stroke ?? 'var(--rule)'} strokeWidth={c.width ?? 1} />
          <text x={X0 + i * C + C / 2 - 1} y={y + H / 2 + 5} fontSize="14" textAnchor="middle" fontWeight={c.bold ? 700 : 400}
            style={{ fill: c.ink ?? 'var(--ink)', fontVariantNumeric: 'tabular-nums' }}>{c.text}</text>
        </g>
      ))}
    </g>
  );
}

function Frame({ children, head, foot }: { children: ReactNode; head: ReactNode; foot?: ReactNode }) {
  return (
    <svg viewBox={`0 0 ${W} 300`} role="img" aria-label="Animation of two loops run by threads in two orders"
      style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <text x={16} y={24} fontSize="13" fontWeight="600">{head}</text>
      {children}
      {foot ? <text x={16} y={288} fontSize="13" fontWeight="600">{foot}</text> : null}
    </svg>
  );
}

const plain = (v: number): CellLook => ({ text: Number.isNaN(v) ? '' : String(v) });
const muted = (v: number): CellLook => ({ text: String(v), ink: 'var(--ink-3)' });

/** One loop under one thread order, `p` of the way through. */
function running(op: Loop, order: OrderName, p: number): ReactNode {
  const ord = threadOrder(order, N);
  const done = Math.min(N, Math.floor(ramp(p, 0.1, 0.9) * (N + 0.999)));
  const state = runThreads(op, DATA, ord.slice(0, done), params(op));
  const last = done > 0 ? ord[done - 1] : null; // the thread that has just run
  const r = last == null ? null : reads(op, last, params(op));
  const want = SCENE_WANT[op];
  const written = new Set(ord.slice(0, done));
  const orderText = ord.map((k, i) => (i === done - 1 ? `[${k}]` : String(k))).join(' ');
  const wrong = [...written].filter((k) => state[k] !== want[k]).length;

  // What the thread that just ran read: its own inputs, and for the in-place loop its left neighbour as it was then.
  const readIdx = new Set<number>();
  if (r) {
    r.inputs.forEach((a) => a.indices.forEach((i) => readIdx.add(i)));
    r.outputs.forEach((i) => readIdx.add(i));
  }
  const result = (k: number): CellLook => {
    if (!written.has(k)) return op === 'inplace' ? plain(SCENE_X[k]) : plain(NaN);
    const ok = state[k] === want[k];
    return {
      text: String(state[k]),
      fill: ok ? 'var(--ok-wash)' : 'var(--fail-wash)',
      stroke: k === last ? 'var(--ink)' : ok ? 'var(--ok)' : 'var(--fail)',
      ink: ok ? 'var(--ink)' : 'var(--fail)',
      bold: k === last,
      width: k === last ? 2.2 : 1,
    };
  };

  const head = (
    <>
      {op === 'neighbour' ? 'Loop 1' : 'Loop 2'}: {OP_BODY[op]}
    </>
  );
  const foot =
    done === 0
      ? `threads will run in this order: ${ord.join(' ')}`
      : done < N
        ? `threads run so far: ${orderText}`
        : wrong === 0
          ? 'all 8 threads done: the same answer as the loop'
          : `all 8 threads done: ${wrong} of 8 results differ from the loop`;

  if (op === 'neighbour')
    return (
      <Frame head={head} foot={foot}>
        <Row y={50} label="x" sub="never written"
          cells={SCENE_X.map((v, i) =>
            readIdx.has(i) ? { text: String(v), fill: 'var(--global)', stroke: 'var(--global)', ink: 'var(--page)', bold: true } : plain(v),
          )} />
        <Row y={128} label="y" sub="one element per thread" cells={SCENE_X.map((_, k) => result(k))} />
        <Row y={206} label="what the loop gives" cells={want.map(muted)} />
      </Frame>
    );

  // In place: the neighbour read is either a value some thread already wrote, or the original.
  const left = last != null && last > 0 ? last - 1 : null;
  const leftWasWritten = left != null && ord.slice(0, done - 1).includes(left);
  return (
    <Frame head={head} foot={foot}>
      <Row y={50} label="x at the start" cells={SCENE_X.map(muted)} />
      <Row y={128} label="x in memory now" sub="read and written"
        cells={SCENE_X.map((_, k) => {
          const c = result(k);
          return k === left ? { ...c, stroke: 'var(--ink)', width: 2.2 } : c;
        })} />
      {left != null && last != null ? (
        <g>
          <path d={`M ${X0 + left * C + C / 2} ${128 + H + 3} q ${C / 2} 16 ${C - 4} 0`} fill="none" stroke="var(--ink)" strokeWidth="1.6" />
          <text x={X0 + left * C + C} y={128 + H + 30} fontSize="11.5" textAnchor="middle">
            {leftWasWritten ? `thread ${last} read what thread ${left} wrote` : `thread ${last} read x[${left}] before thread ${left} ran`}
          </text>
        </g>
      ) : null}
      <Row y={206} label="what the loop gives" cells={want.map(muted)} />
    </Frame>
  );
}

function intro(p: number): ReactNode {
  const show2 = ramp(p, 0.32, 0.5);
  return (
    <Frame head="8 numbers and two loops" foot={p > 0.7 ? 'one thread per value of k: does the order of the threads matter?' : undefined}>
      <Row y={50} label="x" cells={SCENE_X.map(plain)} />
      <g opacity={ramp(p, 0.12, 0.3)}>
        <text x={16} y={140} fontSize="13" fontWeight="600">Loop 1</text>
        <text x={76} y={140} fontSize="13" style={MONO}>for each k: {OP_BODY.neighbour}</text>
        <text x={76} y={160} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>writes a second array, y ({OP_NOTE.neighbour})</text>
      </g>
      <g opacity={show2}>
        <text x={16} y={204} fontSize="13" fontWeight="600">Loop 2</text>
        <text x={76} y={204} fontSize="13" style={MONO}>for each k: {OP_BODY.inplace}</text>
        <text x={76} y={224} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>writes back into x ({OP_NOTE.inplace})</text>
      </g>
    </Frame>
  );
}

function summary(p: number): ReactNode {
  const rev = (op: Loop) => runThreads(op, DATA, threadOrder('reversed', N), params(op));
  const cells = (op: Loop, got: number[]): CellLook[] =>
    got.map((v, k) =>
      v === SCENE_WANT[op][k]
        ? { text: String(v), fill: 'var(--ok-wash)', stroke: 'var(--ok)' }
        : { text: String(v), fill: 'var(--fail-wash)', stroke: 'var(--fail)', ink: 'var(--fail)' },
    );
  const bad = rev('inplace').filter((v, k) => v !== SCENE_WANT.inplace[k]).length;
  return (
    <Frame head="Run backwards, compared with what each loop gives"
      foot={p > 0.5 ? 'the test: does any thread read what another thread writes?' : undefined}>
      <Row y={56} label="loop 1, backwards" sub="reads overlap, nobody writes x" cells={cells('neighbour', rev('neighbour'))} />
      <g opacity={ramp(p, 0.15, 0.3)}>
        <Row y={140} label="loop 2, backwards" sub={`${bad} of 8 differ`} cells={cells('inplace', rev('inplace'))} />
        <Row y={206} label="loop 2, in loop order" cells={SCENE_WANT.inplace.map(muted)} />
      </g>
    </Frame>
  );
}

const beats: Beat[] = [
  { clip: 'order-1', label: 'Two loops', draw: intro },
  { clip: 'order-2', label: 'Loop 1 in order', draw: (p) => running('neighbour', 'forward', p) },
  { clip: 'order-3', label: 'Loop 1 backwards', draw: (p) => running('neighbour', 'reversed', p) },
  { clip: 'order-4', label: 'Loop 2 in order', draw: (p) => running('inplace', 'forward', p) },
  { clip: 'order-5', label: 'Loop 2 backwards', draw: (p) => running('inplace', 'reversed', p) },
  { clip: 'order-6', label: 'The test', draw: summary },
];

export function OrderScene() {
  return <Scene chapter="ch02" title="Same loop, another order" beats={beats} />;
}
