import { useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Simplification, Source } from '../../components/notes';
import { href } from '../../lib/router';
import { CORRECT, makeProblem, naiveBlockReads, phases, runBlock, traffic, type Config, type Step } from '../../sim/tiled';
import { WhyTilingScene } from './WhyTilingScene';

// ---------------------------------------------------------------- drawing
const sum = (a: number[]) => a.reduce((x, y) => x + y, 0);
const plural = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`;

/** Cell geometry shared by M, N and P, so rows and columns line up, with a wider gap between tiles. */
function cellBox(size: number, r: number, c: number, tile: number): CSSProperties {
  return {
    width: size,
    height: size,
    marginLeft: c > 0 && c % tile === 0 ? 5 : 0,
    marginTop: r > 0 && r % tile === 0 ? 5 : 0,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 0,
    borderRadius: 2,
    border: '1px solid var(--rule)',
    background: 'var(--cell)',
    color: 'var(--ink)',
    fontSize: size >= 40 ? '0.76rem' : '0.72rem',
    fontWeight: 450,
    lineHeight: 1.15,
    fontVariantNumeric: 'tabular-nums',
  };
}
const READ: CSSProperties = { background: 'var(--global-wash)', border: '1px solid var(--global)' };
/** A caption above a group of buttons (a <label> would forward clicks on the caption to the first button). */
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };

/** An element name with its subscripts, e.g. M 0,3. */
function El({ a, r, c }: { a: string; r: number; c: number }) {
  return (
    <span className="nowrap">
      {a}
      <sub style={{ fontSize: '0.86em', verticalAlign: '-0.15em', lineHeight: 0 }}>
        {r},{c}
      </sub>
    </span>
  );
}

function Matrix({ name, n, note, children }: { name: string; n: number; note?: string; children: ReactNode }) {
  return (
    <figure style={{ margin: 0 }}>
      <figcaption style={{ marginBottom: 4 }}>
        <b className="t-global">{name}</b>
        {note ? <span className="muted"> {note}</span> : null}
      </figcaption>
      <div
        role="group"
        aria-label={`matrix ${name}`}
        style={{ display: 'grid', gridTemplateColumns: `repeat(${n}, max-content)`, gap: 2, justifyContent: 'start' }}
      >
        {children}
      </div>
    </figure>
  );
}

const cells = (n: number) => Array.from({ length: n * n }, (_, i) => ({ i, r: Math.floor(i / n), c: i % n }));

// ---------------------------------------------------------------- interactive 1: who reads what
const W_CHOICES = [4, 6, 8];
const B_CHOICES = [2, 3, 4];

function WhoReadsWhat() {
  const [w, setW] = useState(4);
  const [b, setB] = useState(2);
  const [sel, setSel] = useState({ r: 0, c: 0 });
  const [view, setView] = useState<'thread' | 'block'>('thread');
  const pickW = (nw: number) => {
    setW(nw);
    if (nw % b !== 0) setB(B_CHOICES.find((x) => nw % x === 0)!);
    setSel((s) => ({ r: Math.min(s.r, nw - 1), c: Math.min(s.c, nw - 1) }));
  };
  const bx = Math.floor(sel.c / b);
  const by = Math.floor(sel.r / b);
  // One thread is a block of size 1 at its own position: the same count, no second code path.
  const reads = useMemo(
    () => (view === 'thread' ? naiveBlockReads(w, 1, sel.c, sel.r) : naiveBlockReads(w, b, bx, by)),
    [view, w, b, sel.r, sel.c, bx, by],
  );
  const loads = sum(reads.M) + sum(reads.N);
  const distinct = reads.M.filter((x) => x > 0).length + reads.N.filter((x) => x > 0).length;
  const ratio = loads / distinct;
  const size = w <= 4 ? 46 : w <= 6 ? 40 : 36;

  const input = (a: 'M' | 'N', counts: number[]) =>
    cells(w).map(({ i, r, c }) => (
      <div
        key={i}
        data-cell={`${a}-${r}-${c}`}
        data-count={counts[i]}
        style={{ ...cellBox(size, r, c, b), ...(counts[i] ? READ : null) }}
      >
        <El a={a} r={r} c={c} />
        {view === 'block' && counts[i] ? (
          <b className="t-global" style={{ marginTop: 2 }}>
            ×{counts[i]}
          </b>
        ) : null}
      </div>
    ));

  return (
    <Bench
      title="Who reads what"
      task="Click an element of P. Then switch to its whole block and read off how often each input element is fetched."
      foot="Counts are loads the naive kernel requests from global memory: one element of M and one of N per loop iteration, per thread."
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span style={FIELD}>
          Width W = {w}
          <span className="seg" role="group" aria-label="Width W">
            {W_CHOICES.map((x) => (
              <button key={x} aria-pressed={w === x} onClick={() => pickW(x)}>
                {x}
              </button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          Block size = {b} × {b}
          <span className="seg" role="group" aria-label="Block size">
            {B_CHOICES.map((x) => (
              <button key={x} aria-pressed={b === x} disabled={w % x !== 0} onClick={() => setB(x)}>
                {x}
              </button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          Show the reads of
          <span className="seg" role="group" aria-label="Show the reads of">
            <button aria-pressed={view === 'thread'} onClick={() => setView('thread')}>
              one thread
            </button>
            <button aria-pressed={view === 'block'} onClick={() => setView('block')}>
              its whole block
            </button>
          </span>
        </span>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.2rem', alignItems: 'flex-start' }}>
        <div style={{ overflowX: 'auto', maxWidth: '100%', paddingBottom: 2 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'max-content max-content', gap: 14 }}>
            <div style={{ alignSelf: 'end', maxWidth: size * w, color: 'var(--ink-2)', fontSize: '0.78rem' }}>
              <span className="legend">
                <span>
                  <i style={{ background: 'var(--global-wash)', border: '1px solid var(--global)' }} />
                  loaded from global memory
                </span>
              </span>
            </div>
            <Matrix name="N" n={w}>
              {input('N', reads.N)}
            </Matrix>
            <Matrix name="M" n={w}>
              {input('M', reads.M)}
            </Matrix>
            <Matrix name="P" n={w} note="(click an element)">
              {cells(w).map(({ i, r, c }) => {
                const inBlock = Math.floor(r / b) === by && Math.floor(c / b) === bx;
                const on = view === 'thread' ? r === sel.r && c === sel.c : inBlock;
                return (
                  <button
                    key={i}
                    aria-label={`P[${r}][${c}]`}
                    aria-pressed={on}
                    onClick={() => setSel({ r, c })}
                    style={{
                      ...cellBox(size, r, c, b),
                      cursor: 'pointer',
                      ...(on ? { background: 'var(--global-wash)', border: '2px solid var(--ink)' } : null),
                    }}
                  >
                    <El a="P" r={r} c={c} />
                  </button>
                );
              })}
            </Matrix>
          </div>
        </div>
        <div className="panel" style={{ flex: '1 1 15rem', minWidth: '14rem', maxWidth: '30rem' }} aria-live="polite">
          {view === 'thread' ? (
            <p style={{ margin: 0 }} data-readout="who">
              <b>
                The thread for P[{sel.r}][{sel.c}]
              </b>{' '}
              reads row {sel.r} of M and column {sel.c} of N.
            </p>
          ) : (
            <p style={{ margin: 0 }} data-readout="who">
              <b>
                Block (x {bx}, y {by})
              </b>
              : the {b * b} threads for rows {by * b} to {by * b + b - 1} and columns {bx * b} to {bx * b + b - 1} of P,
              each running the naive kernel.
            </p>
          )}
          <table className="data" style={{ margin: '0.6rem 0' }}>
            <tbody>
              <tr>
                <td>Loads from global memory</td>
                <td className="num" data-total="loads">
                  {loads}
                </td>
              </tr>
              <tr>
                <td>Distinct elements needed</td>
                <td className="num" data-total="distinct">
                  {distinct}
                </td>
              </tr>
              <tr>
                <td>Loads per element</td>
                <td className="num" data-total="ratio">
                  <b>{Number.isInteger(ratio) ? ratio : ratio.toFixed(2)}</b>
                </td>
              </tr>
            </tbody>
          </table>
          {view === 'thread' ? (
            <p className="muted" style={{ margin: 0 }}>
              On its own a thread wastes nothing: it needs each of these elements once and loads each once. The waste
              only shows when you look at its neighbours. Switch to the whole block.
            </p>
          ) : (
            <p className="muted" style={{ margin: 0 }}>
              The number on an element is how many threads of this block load it. A block that loaded each element once
              and shared it would issue {distinct} loads instead of {loads}.
            </p>
          )}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- interactive 2: phases
const PHASE_CASES = [
  { w: 4, t: 2 },
  { w: 6, t: 2 },
  { w: 6, t: 3 },
];
const isLoad = (s: Step) => s.op === 'loadM' || s.op === 'loadN';

function PhasesBench() {
  const [ci, setCi] = useState(0);
  const [ph, setPh] = useState(0);
  const { w, t } = PHASE_CASES[ci];
  const { prob, nPh, steps } = useMemo(() => {
    const cfg: Config = { ...CORRECT, width: w, tile: t };
    const prob = makeProblem(w);
    return { prob, nPh: phases(cfg), steps: runBlock(cfg, prob, 0, 0, 'lockstep').steps };
  }, [w, t]);

  // Everything shown is read off the simulator's trace of block (0,0).
  const loads = steps.filter((s) => isLoad(s) && s.ph === ph);
  const tile = (a: 'M' | 'N') => new Set(loads.filter((s) => s.global!.array === a).map((s) => s.global!.index));
  const threads = Array.from({ length: t * t }, (_, i) => {
    const mine = steps.filter((s) => s.thread === i);
    const muls = mine.filter((s) => s.op === 'mul' && s.ph === ph);
    const earlier = mine.filter((s) => s.op === 'mul' && s.ph < ph);
    return {
      m: mine.find((s) => s.op === 'loadM' && s.ph === ph)!,
      n: mine.find((s) => s.op === 'loadN' && s.ph === ph)!,
      muls,
      before: earlier.length ? earlier[earlier.length - 1].pvalue! : 0,
      after: muls[muls.length - 1].pvalue!,
      store: mine.find((s) => s.op === 'store')!,
    };
  });
  const writes = (a: 'Mds' | 'Nds', r: number, c: number) =>
    steps.filter((s) => s.write && s.write.array === a && s.write.r === r && s.write.c === c && s.ph <= ph);
  const last = ph === nPh - 1;
  const gsize = w <= 4 ? 40 : 34;

  const global = (a: 'M' | 'N', data: number[]) => {
    const on = tile(a);
    return (
      <Matrix name={a} n={w}>
        {cells(w).map(({ i, r, c }) => (
          <div
            key={i}
            data-cell={`${a}-${r}-${c}`}
            data-in-tile={on.has(i) ? 1 : 0}
            style={{ ...cellBox(gsize, r, c, t), ...(on.has(i) ? READ : { color: 'var(--ink-2)' }) }}
          >
            <span style={{ fontSize: '0.86em' }} className="muted">
              {r},{c}
            </span>
            <b>{data[i]}</b>
          </div>
        ))}
      </Matrix>
    );
  };

  const shared = (a: 'Mds' | 'Nds') => (
    <figure style={{ margin: 0 }}>
      <figcaption style={{ marginBottom: 4 }}>
        <b className="t-shared">{a}</b>
      </figcaption>
      <div
        role="group"
        aria-label={`shared tile ${a}`}
        style={{ display: 'grid', gridTemplateColumns: `repeat(${t}, max-content)`, gap: 2 }}
      >
        {cells(t).map(({ i, r, c }) => {
          const ws = writes(a, r, c);
          const now = ws[ws.length - 1];
          const g = now.global!;
          return (
            <div
              key={i}
              data-slot={`${a}-${r}-${c}`}
              data-value={now.write!.value}
              data-writes={ws.length}
              style={{
                width: 88,
                height: 76,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                border: '1px solid var(--shared)',
                background: 'var(--shared-wash)',
                borderRadius: 2,
                fontSize: '0.7rem',
                lineHeight: 1.2,
                fontVariantNumeric: 'tabular-nums',
              }}
            >
              <span className="muted">
                {a}[{r}][{c}]
              </span>
              <b className="t-shared" style={{ fontSize: '1.15rem', lineHeight: 1.15 }}>
                {now.write!.value}
              </b>
              <span className="t-global">
                from {g.array}[{g.row}][{g.col}]
              </span>
              <span className="muted" style={{ minHeight: '1.2em' }}>
                {ws.length > 1 ? (
                  <>
                    was{' '}
                    {ws.slice(0, -1).map((s, k) => (
                      <s key={k} style={{ marginRight: 3 }}>
                        {s.write!.value}
                      </s>
                    ))}
                  </>
                ) : null}
              </span>
            </div>
          );
        })}
      </div>
    </figure>
  );

  return (
    <Bench
      title="Phases"
      task="Step through the phases of block (0,0). Watch where each thread's two loads go, and what happens to the shared slots."
      foot={
        <>
          Inputs are small integers, so every sum is exact. All {t * t} threads are shown moving together; what keeps
          real threads in step is the subject of section 5.4.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Problem
          <span className="seg" role="group" aria-label="Problem">
            {PHASE_CASES.map((p, i) => (
              <button
                key={i}
                aria-pressed={ci === i}
                onClick={() => {
                  setCi(i);
                  setPh(0);
                }}
              >
                {p.w}×{p.w}, tile {p.t}
              </button>
            ))}
          </span>
        </span>
        <button onClick={() => setPh(ph - 1)} disabled={ph === 0}>
          Previous phase
        </button>
        <button className="primary" onClick={() => setPh(ph + 1)} disabled={last}>
          Next phase
        </button>
        <span data-readout="phase" aria-live="polite" style={{ paddingBottom: '0.36rem' }}>
          <b>Phase {ph}</b> of {nPh} (numbered 0 to {nPh - 1}){last ? ': the last one' : ''}
        </span>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.2rem', alignItems: 'stretch', marginBottom: '1.2rem' }}>
        <div className="panel" style={{ flex: '0 1 auto', maxWidth: '100%', overflowX: 'auto' }}>
          <p style={{ margin: '0 0 0.6rem' }}>
            <b>Global memory.</b> The two tiles this phase loads.
          </p>
          <div style={{ display: 'flex', gap: '1.2rem' }}>
            {global('M', prob.M)}
            {global('N', prob.N)}
          </div>
        </div>
        <div className="panel" style={{ flex: '0 1 auto', maxWidth: '100%', overflowX: 'auto' }}>
          <p style={{ margin: '0 0 0.6rem' }}>
            <b>Shared memory of block (0,0).</b> The same {plural(2 * t * t, 'slot')} in every phase.
          </p>
          <div style={{ display: 'flex', gap: '1.2rem' }}>
            {shared('Mds')}
            {shared('Nds')}
          </div>
          <p className="muted" style={{ margin: '0.6rem 0 0' }} data-readout="overwrites">
            {ph === 0
              ? 'Each slot has been written once.'
              : `Each slot has now been written ${ph + 1} times. The crossed-out values are what earlier phases left; they are gone.`}
          </p>
        </div>
      </div>
      <div className="panel" style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }}>
          <thead>
            <tr>
              <th>Thread</th>
              <th>Loads from M</th>
              <th>Loads from N</th>
              <th>Adds in phase {ph}</th>
              <th className="num">Running sum</th>
            </tr>
          </thead>
          <tbody>
            {threads.map((th) => {
              const slotOf = (s: Step) => `${s.write!.array}[${s.write!.r}][${s.write!.c}]`;
              const elOf = (s: Step) => `${s.global!.array}[${s.global!.row}][${s.global!.col}]`;
              return (
                <tr key={th.m.thread} data-thread={`${th.m.tx}-${th.m.ty}`}>
                  <td className="nowrap">
                    tx {th.m.tx}, ty {th.m.ty}
                  </td>
                  <td className="nowrap" data-col="m">
                    <span className="t-global">{elOf(th.m)}</span> → <span className="t-shared">{slotOf(th.m)}</span>
                  </td>
                  <td className="nowrap" data-col="n">
                    <span className="t-global">{elOf(th.n)}</span> → <span className="t-shared">{slotOf(th.n)}</span>
                  </td>
                  <td className="nowrap" data-col="adds">
                    {th.muls.map((s) => `${s.reads![0].value}×${s.reads![1].value}`).join(' + ')}
                  </td>
                  <td className="num nowrap" data-col="sum">
                    {th.before} + {th.after - th.before} = <span className="chip reg">{th.after}</span>
                    {last ? (
                      <>
                        {' '}
                        = <span className="t-global">{elOf(th.store)}</span>
                      </>
                    ) : null}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
// Numbers come from the simulator's traffic model, not from arithmetic done by hand.
const NAIVE_96 = traffic(96, null);
const TILED_96 = traffic(96, 16);
const NAIVE_8 = traffic(8, null);
const TILED_8 = traffic(8, 4);

export const questionsS53: Record<string, Question> = {
  'ch05/5.3/q-requests': {
    kind: 'numeric',
    id: 'ch05/5.3/q-requests',
    prompt: (
      <>
        Two 96×96 matrices are multiplied, first by the naive kernel and then by a tiled kernel with 16×16 tiles. Follow
        one particular element of M through the whole multiplication. Work out how many times the naive kernel requests
        it from global memory; then answer: how many times does the tiled kernel request it?
      </>
    ),
    answer: TILED_96.perElement,
    unit: 'requests',
    explain: (
      <p>
        An element in row r of M is needed by the {NAIVE_96.perElement} threads that compute row r of P, and in the
        naive kernel each of them fetches it for itself: {NAIVE_96.perElement} requests. Those threads sit in 96 / 16 ={' '}
        {TILED_96.perElement} blocks. A tiled block fetches the element once, in the one phase whose tile contains it,
        and all 16 of its threads in that row use the copy. So {TILED_96.perElement} requests: N without tiling, N/T
        with T×T tiles.
      </p>
    ),
    source: 'ch. 5 exercise 8, p. 119; §5.3, p. 105',
  },
  'ch05/5.3/q-addition': {
    kind: 'choice',
    id: 'ch05/5.3/q-addition',
    prompt: (
      <>
        Matrix addition, <code>C[i][j] = A[i][j] + B[i][j]</code>, one thread per output element. Could loading tiles of
        A and B into shared memory cut its global-memory traffic the way it does for multiplication?
      </>
    ),
    options: [
      {
        label: 'Yes, by a factor of the tile width, as for multiplication.',
        why: 'That factor is the number of threads in a block that need the same element. Count them here.',
      },
      {
        label: 'Yes, but only by a factor of 2, because there are two input matrices.',
        why: 'The number of inputs is not what tiling exploits. Several threads needing the same element is.',
      },
      {
        label:
          'No. Each element of A and B is needed by exactly one thread, once, so there is no repeated fetch to remove.',
        correct: true,
      },
      {
        label: 'No, because the tiles of an addition are too large for shared memory.',
        why: 'A tile is whatever size you choose. Size is not the obstacle; the absence of reuse is.',
      },
    ],
    explain: (
      <p>
        Tiling does not make a fetch cheaper. It replaces several fetches of the same element by one fetch and several
        reads of a shared copy. In addition every input element is already fetched exactly once, by the only thread
        that needs it, so the traffic stays at two loads and one store per operation and a detour through shared memory
        would only add work. The question to ask of any kernel before reaching for shared memory: which elements does
        more than one thread of a block need?
      </p>
    ),
    source: 'ch. 5 exercise 1, p. 119; §5.1, p. 96',
  },
  'ch05/5.3/q-apply': {
    kind: 'numeric',
    id: 'ch05/5.3/q-apply',
    prompt: (
      <>
        The explorer counted one block. Now count a whole multiplication: 8×8 matrices, 4×4 tiles, every block, every
        phase. How many loads of input elements does the tiled kernel request from global memory in total? Work out the
        naive kernel's total too; the explanation compares them.
      </>
    ),
    answer: TILED_8.loads,
    unit: 'loads',
    explain: (
      <p>
        Tiled: 4 blocks × 2 phases × 16 threads × 2 loads = {TILED_8.loads}. Naive: 64 threads × 8 iterations × 2 loads
        = {NAIVE_8.loads}. The ratio is {NAIVE_8.loads / TILED_8.loads}, the tile width. Put another way, the two
        inputs hold 128 elements; the naive kernel requests each one {NAIVE_8.perElement} times and the tiled kernel{' '}
        {TILED_8.perElement} times. With 2×2 tiles on the same matrices the total would be {traffic(8, 2).loads}.
      </p>
    ),
    source: 'ch. 5 exercise 2, p. 119; §5.3, pp. 105–107',
  },
};

// ---------------------------------------------------------------- the section
export function S53() {
  return (
    <>
      <p>
        Global memory is large and slow. Shared memory is fast and small. The naive kernel uses only the first, and
        section 5.1 put a number on the cost: two loads from global memory for every multiply and add. Tiling is the
        standard way to use both memories for what each is good at. Cut the data into pieces, called <b>tiles</b>, each
        small enough to fit in shared memory. Bring one tile in, do all the work that needs it, move on to the next.
      </p>
      <Source section="5.3" pages="104" />
      <p>
        The condition is hidden in "all the work that needs it". Tiling only applies when the computation can be
        arranged so that the work on one tile does not need the others at the same moment, and it only pays when the
        data in a tile is used more than once while it is there. Not every kernel has that shape. Matrix multiplication
        does, and the first step is to see where.
      </p>

      <h2>Who reads what</h2>
      <p>
        Take the book's small case: 4×4 matrices, with the threads grouped into 2×2 blocks. Block (0,0) holds the four
        threads that compute P<sub>0,0</sub>, P<sub>0,1</sub>, P<sub>1,0</sub> and P<sub>1,1</sub>. The thread for P
        <sub>0,0</sub> walks along row 0 of M and down column 0 of N. The thread for P<sub>0,1</sub> walks along row 0
        of M again, and down column 1 of N. Two threads in the same row of a block read the same row of M; two threads
        in the same column read the same column of N. Nothing in the naive kernel lets one of them use what the other
        has just fetched.
      </p>
      <Source section="5.3" pages="104–105" figs="Figs. 5.5 and 5.6" />
      <Predict
        id="ch05/5.3/p-overlap"
        prompt="Scale that up: 8×8 matrices, 4×4 blocks, naive kernel. How many times does one block fetch each element of M that it needs?"
        unit="times"
      >
        {(v) => (
          <p>
            <b>Four.</b> A block's 16 threads cover 4 rows of P, and the 4 threads that share a row each read that
            whole row of M.{' '}
            {v === 4
              ? 'That is what you had.'
              : v === 16
                ? 'Sixteen would mean every thread of the block reads every element. Only the threads in the same row of P read a given row of M.'
                : v === 8
                  ? 'Eight is how many elements of M one thread loads, not how many threads load one element.'
                  : 'Count the threads of the block that sit in one row of P.'}{' '}
            Set the explorer to width 8 and block size 4 and check it.
          </p>
        )}
      </Predict>
      <WhoReadsWhat />
      <p>
        Whatever you select, the block's loads divided by the elements it needs comes out as the width of the block.
        That is the general result. In a T×T block, T threads ask for each element, so a block that fetched each
        element once and shared it would cut its requests to global memory by a factor of T. With the 2×2 blocks of the
        book's example the traffic halves; with 32×32 blocks it falls to one thirty-second.
      </p>
      <Source section="5.3" pages="105" />
      <p>
        The same factor holds for the multiplication as a whole. In the naive kernel each input element is requested
        once by every thread that uses it: W times for W×W matrices. Shared inside T×T blocks, it would be requested
        once by every block that uses it: W/T times. Keep in mind what is being counted. These are requests, the same
        thing <a href={href('ch05', '5.1')}>section 5.1</a> counted when it charged every load to global memory, and a
        request is not always a trip to off-chip memory.
      </p>
      <Simplification>
        <p>
          The counts are what the kernel <em>asks</em> of global memory in the book's model, which has no caches. Where
          on-chip caches already absorb repeated reads, the measured speed-up from tiling is smaller than T. Lab 5.2
          measures it on your device.
        </p>
      </Simplification>
      <Ask q={questionsS53['ch05/5.3/q-requests']} />

      <h2>From sharing to phases</h2>
      <p>
        How does a block arrange to fetch each element once? It cannot simply copy everything it needs into shared
        memory: that is T whole rows of M and T whole columns of N, and it grows with the matrix. So the work is cut
        along the dot product instead. Divide M and N into T×T tiles, the same size as a block. Each thread's dot
        product of W terms becomes W/T partial sums of T terms each, and the block computes them one <b>phase</b> at a
        time. A phase has two steps:
      </p>
      <Source section="5.3" pages="105–107" figs="Figs. 5.7 and 5.8" />
      <ol>
        <li>
          Every thread loads one element of M and one element of N from global memory into two shared arrays,{' '}
          <code>Mds</code> and <code>Nds</code>. Between them the T² threads bring in one whole tile of each.
        </li>
        <li>
          Every thread adds T products to its running sum, reading only <code>Mds</code> and <code>Nds</code>.
        </li>
      </ol>
      <p>
        In phase 0 the block loads the tile at the left end of its rows of M and the tile at the top of its columns of
        N. Each loaded value is then read T times: an element of <code>Mds</code> by the T threads in its row of the
        block, an element of <code>Nds</code> by the T threads in its column. Phase 1 moves one tile to the right in M
        and one tile down in N, and so on for W/T phases. One request to global memory, T uses.
      </p>
      <Aside>
        <p>
          Phases are numbered from 0 here, like the loop variable <code>ph</code> in the kernel. The book's Fig. 5.8
          does the same, while its text on p. 107 calls the same two phases 1 and 2.
        </p>
      </Aside>
      <WhyTilingScene />
      <p>
        The stepper runs block (0,0) of a small multiplication with real numbers. For each phase it shows the two
        tiles being read from global memory, what every thread loads and into which slot, the contents of the two
        shared tiles, and each thread's running sum once the phase is over. Step forward and keep an eye on the shared
        slots.
      </p>
      <Simplification>
        <p>
          The stepper shows whole phases, with all threads of the block moving together. Real threads do not move
          together, and nothing here shows timing or caches. Section 5.4 deals with what that changes.
        </p>
      </Simplification>
      <PhasesBench />

      <h2>Why a small memory is enough</h2>
      <p>
        Two things in the stepper carry the idea. First, <code>Mds</code> and <code>Nds</code> never grow. Each phase
        writes into the very slots the previous phase used, because by then every thread has taken what it needed from
        them. Two tiles of T² values serve a multiplication of any width, which is how a small shared memory ends up
        answering most of the kernel's reads.
      </p>
      <Source section="5.3" pages="107" />
      <p>
        Second, in each phase all the threads of the block work on the same small piece of the input. That property is
        called <b>locality</b>, and it is the reason a small fast memory can stand in for a large slow one. It is not
        special to GPUs: on a multicore CPU the small fast memory is the cache, and the same restructuring helps there.
      </p>
      <p>
        Reusing the slots has a price that the stepper hides by moving all threads at once. No thread may read a slot
        before its neighbour has loaded it, and no thread may load the next phase's value while a neighbour still needs
        the old one. <a href={href('ch05', '5.4')}>Section 5.4</a> writes the kernel and pays that price.
      </p>
      <Ask q={questionsS53['ch05/5.3/q-addition']} />
      <Ask q={questionsS53['ch05/5.3/q-apply']} />
    </>
  );
}
