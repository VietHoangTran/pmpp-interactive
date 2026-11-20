import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { href } from '../../lib/router';
import { BOOK, EX1, backtrace, inNeighbours, levels, showLevel, storage, toCOO, toCSC, toCSR, treeEdges, UNVISITED } from '../../sim/bfs';
import { BfsGraph } from './BfsGraph';

// Before you start (book pp. 425-431, §18.1 and §18.2, plus the chapter's purpose from
// pp. 425, 445 and 449). What the book says comes from notes/ch18.md; vendor facts from
// notes/today/ch18-facts.md (D3, D5, D6). Prerequisite questions draw on the notes of
// chapters 4, 9, 11, 12 and 17. Models: src/sim/bfs.ts.

export const questionsS180: Record<string, Question> = {
  'ch18/start/q-csr-read': {
    kind: 'choice',
    id: 'ch18/start/q-csr-read',
    prompt: (
      <>
        <code>srcPtrs = [0,2,4,7,9,11,12,13,15,15]</code> and <code>dst = [1,2,3,4,5,6,7,4,8,5,8,6,8,0,6]</code>. Which
        edges leave vertex 4, and how many leave vertex 8?
      </>
    ),
    options: [
      { label: '4 → 5 and 4 → 8; none leave 8.', correct: true },
      {
        label: '4 → 4 and 4 → 8; one leaves 8.',
        why: 'That reads dst at srcPtrs[3] = 7 and onwards, the slice of vertex 3. Vertex v\'s slice starts at srcPtrs[v].',
      },
      {
        label: '4 → 5 and 4 → 8; two leave 8, to 0 and 6.',
        why: 'dst[13] and dst[14] belong to vertex 7: srcPtrs[7] = 13 and srcPtrs[8] = 15.',
      },
      {
        label: 'Only 4 → 5: the slice is srcPtrs[4] = 9 to srcPtrs[5] = 11, and the end is excluded, so one entry.',
        why: 'From 9 up to, not including, 11 is two entries: dst[9] and dst[10].',
      },
    ],
    explain: (
      <p>
        A vertex's out-edges are the slice of <code>dst</code> from <code>srcPtrs[v]</code> up to, not including,{' '}
        <code>srcPtrs[v+1]</code>. For 4 that is <code>dst[9..10]</code> = 5, 8. For 8 it is from 15 to 15: an empty
        slice, so 8 has no out-edges. The extra last entry of <code>srcPtrs</code> exists so this rule works for the last
        vertex too. This is Exercise 1b's method.
      </p>
    ),
    source: 'ch. 18 §18.1, p. 427, Fig. 18.3',
  },
  'ch18/start/q-csc-in': {
    kind: 'choice',
    id: 'ch18/start/q-csc-in',
    prompt: (
      <>
        A pull kernel needs every vertex that points to vertex 6 in the book's graph. Which arrays give that list as
        one contiguous slice, and what is in it?
      </>
    ),
    options: [
      { label: 'CSC: dstPtrs[6] = 8 to dstPtrs[7] = 11, so src[8..10] = 2, 5, 7.', correct: true },
      {
        label: 'CSR: srcPtrs[6] = 12 to srcPtrs[7] = 13, so dst[12] = 8.',
        why: 'That is where 6 points (6 → 8), not who points at 6.',
      },
      {
        label: 'COO: the entries whose dst is 6 sit next to each other.',
        why: 'The book\'s COO is sorted by source, so the entries with dst 6 are scattered: you scan all 15 to find them.',
      },
    ],
    explain: (
      <p>
        CSC groups edges by destination, so a column is one slice. With CSR you would search every row for a 6; with
        COO you would scan all 15 edges. Which format you keep decides which question is cheap, and so which kernel
        you can write.
      </p>
    ),
    source: 'ch. 18 §18.1, pp. 428–429, Fig. 18.3',
  },
  'ch18/start/q-path': {
    kind: 'choice',
    id: 'ch18/start/q-path',
    prompt: <>With root 2, back-trace from vertex 1. Which chain is a shortest path?</>,
    options: [
      { label: '1 ← 0 ← 7 ← 2', correct: true },
      {
        label: '1 ← 0 ← 7 ← 6 ← 2',
        why: '6 and 7 are both at level 1, and the edge between them runs 7 → 6. A step must go to an in-neighbour exactly one level lower.',
      },
      {
        label: '1 ← 4 ← 5 ← 2',
        why: '1 → 4 is an edge leaving 1, so 4 is a child of 1, not a parent. Back-tracing follows incoming edges.',
      },
      {
        label: '1 ← 0 ← 2',
        why: 'There is no edge 2 → 0; 0 is at level 2 with this root.',
      },
    ],
    explain: (
      <p>
        Each step goes to an in-neighbour one level lower: 1's only in-neighbour is 0 (level 2), 0's is 7 (level 1),
        7's is 2. So the path is 2 → 7 → 0 → 1. The walk reads incoming edges, so it needs CSC, or a predecessor
        recorded for each vertex during the search.
      </p>
    ),
    source: 'ch. 18 §18.2, p. 430, Fig. 18.4(b)',
  },
  'ch18/start/q-divergence': {
    kind: 'numeric',
    id: 'ch18/start/q-divergence',
    prompt: (
      <>
        In one warp, lane 0 runs a loop 100 times and every other lane runs it once. How many times does the warp step
        through the loop body?
      </>
    ),
    answer: 100,
    unit: 'passes',
    explain: (
      <p>
        A warp keeps going until its longest-running lane is done. After the first pass every lane but one is idle,
        and those idle lanes still occupy the hardware. On the next page a vertex with many edges does exactly this to
        its warp, and the edge-centric kernel is one way out.
      </p>
    ),
    source: 'ch. 4 §4.5, p. 81 (loops with different trip counts, Fig. 4.10)',
  },
  'ch18/start/q-resident': {
    kind: 'choice',
    id: 'ch18/start/q-resident',
    prompt: (
      <>
        A grid of 64 blocks runs on a device that holds 8 blocks at a time. Every block sets a flag, then spins until
        all 64 flags are set. What happens?
      </>
    ),
    options: [
      {
        label: 'It can hang: the 8 resident blocks wait for blocks that cannot start until a resident block finishes.',
        correct: true,
      },
      {
        label: 'It finishes, more slowly: the hardware swaps the spinning blocks out to let the others in.',
        why: 'Blocks are admitted as earlier ones finish, not swapped out while they run. Nothing here finishes.',
      },
      {
        label: 'It finishes at full speed, because spinning costs nothing.',
        why: 'Speed is not the issue. The 56 blocks that never start cannot set their flags.',
      },
    ],
    explain: (
      <p>
        The runtime keeps a list of blocks still to run and assigns one when a running block finishes. A block that
        waits for a block that has not started can wait for ever. Chapter 11's single-pass scan avoided this by taking
        block indices in the order blocks actually start. Section 18.7 needs a barrier across the whole grid, and
        meets the same problem head on.
      </p>
    ),
    source: 'ch. 4 §4.2, p. 69; ch. 11 §11.9, p. 280',
  },
  'ch18/start/q-workeff': {
    kind: 'choice',
    id: 'ch18/start/q-workeff',
    prompt: (
      <>
        A parallel scan finishes in log₂N steps but performs about N·log₂N additions, where the sequential loop does N
        − 1. In chapter 11's terms, what is wrong with it, and when does that cost you?
      </>
    ),
    options: [
      {
        label: 'It is not work-efficient. The extra additions cost time when there are fewer execution units than the work needs, and energy always.',
        correct: true,
      },
      {
        label: 'Nothing: with enough threads, fewer steps always means a faster kernel, whatever the additions.',
        why: 'Only with enough execution units. With few, the extra additions are paid for in time.',
      },
      {
        label: 'It is not work-efficient, which matters on a CPU with few cores and not on a GPU.',
        why: 'A GPU has many units but not unlimited ones, and every extra operation costs energy.',
      },
    ],
    explain: (
      <p>
        Work efficiency compares the total work with the best sequential algorithm's. The BFS kernels of the next
        page all finish a level in one launch, and most of them are not work-efficient: they check every vertex, or
        every edge, at every level. Section 18.5 counts the cost and fixes it.
      </p>
    ),
    source: 'ch. 11 §11.5, pp. 265–267',
  },
  'ch18/start/q-slot': {
    kind: 'choice',
    id: 'ch18/start/q-slot',
    prompt: (
      <>
        Every passing thread runs <code>i = counter.fetch_add(1, relaxed); out[i] = key;</code>. Two runs on the same
        input give outputs in different orders. Why is no slot ever written twice or skipped?
      </>
    ),
    options: [
      {
        label: 'Each fetch_add returns a different old value, 0, 1, 2 and so on, in whatever order the atomics happen.',
        correct: true,
      },
      {
        label: 'Relaxed ordering makes the threads take their turns in thread order, so each gets the next slot.',
        why: 'Relaxed ordering promises nothing about order. The different output orders show that.',
      },
      {
        label: 'Slots can be written twice when two threads collide; these two runs happened not to collide.',
        why: 'Two fetch_adds on one counter never return the same value: that is what makes them atomic.',
      },
    ],
    explain: (
      <p>
        The set of slots and the final count are fixed; which thread gets which slot is not. That is chapter 12's
        unstable filter, and it is exactly how a BFS frontier is built on the next page.
      </p>
    ),
    source: 'ch. 12 §12.2, pp. 290–291',
  },
  'ch18/start/q-scope': {
    kind: 'choice',
    id: 'ch18/start/q-scope',
    prompt: (
      <>
        A block's private counter lives in shared memory and only that block touches it. The block then adds its count
        to a counter in global memory that every block adds to. Which scopes are right?
      </>
    ),
    options: [
      { label: 'Block scope for the shared counter, device scope for the global one.', correct: true },
      {
        label: 'Block scope for both: each block makes only one update to the global counter.',
        why: 'Other blocks update the same global counter. A block-scope atomic on an object other blocks touch is a data race.',
      },
      {
        label: 'Device scope for both: block scope is never correct.',
        why: 'Device scope is correct for the shared counter but asks for more than it needs; block scope is the right fit there.',
      },
    ],
    explain: (
      <p>
        The scope names the set of threads that synchronise through the atomic. Section 18.6 uses exactly this pair: a
        block-scope counter for the private queue and one device-scope <code>fetch_add</code> per block on the public
        one.
      </p>
    ),
    source: 'ch. 9 §9.4, pp. 212–213; notes/today/ch09-facts.md A2',
  },
  'ch18/start/q-rowthread': {
    kind: 'choice',
    id: 'ch18/start/q-rowthread',
    prompt: (
      <>
        Sparse matrix–vector multiplication: compare one thread per CSR row with one thread per COO non-zero, on a
        matrix where one row holds 10,000 non-zeros and every other row about 3. Which version suffers load imbalance,
        and what does the other pay instead?
      </>
    ),
    options: [
      {
        label: 'One thread per row is imbalanced; one thread per non-zero balances the work but needs atomics on y.',
        correct: true,
      },
      {
        label: 'One thread per non-zero is imbalanced, since the long row\'s non-zeros pile up; one thread per row pays with atomics on y.',
        why: 'Each non-zero is the same amount of work. The row with 10,000 non-zeros is the long job.',
      },
      {
        label: 'Neither: the hardware balances the work across threads.',
        why: 'A thread does the work its index gives it; nothing redistributes it.',
      },
    ],
    explain: (
      <p>
        The thread that owns the long row runs 10,000 iterations while its warp mates wait. One thread per non-zero
        gives every thread one multiply, but several threads now add into the same <code>y[row]</code>. The next page
        makes the same trade between vertex-centric and edge-centric BFS.
      </p>
    ),
    source: 'ch. 17 §17.2 and §17.3, pp. 404–409; ch. 6 on re-mapping threads to work (cited by ch. 18, p. 438)',
  },
};

// ---------------------------------------------------------------- the format lens
type Pick = { kind: 'vertex'; v: number } | { kind: 'edge'; e: number } | null;

function ArrayRow({ id, name, values, hot, on }: { id: string; name: string; values: number[]; hot: Set<number>; on: Set<number> }) {
  return (
    <div data-array={id} style={{ marginBottom: '0.55rem' }}>
      <div className="mx-label t-global">{name}</div>
      <div className="mx global" style={{ gridTemplateColumns: `repeat(${values.length}, auto)` }}>
        {values.map((x, i) => (
          <span key={i} data-i={i} className={hot.has(i) ? 'hot' : on.has(i) ? 'on' : ''} style={{ minWidth: '1.9rem', height: '1.8rem' }}>
            {x}
            <sub>{i}</sub>
          </span>
        ))}
      </div>
    </div>
  );
}

const range = (a: number, b: number) => new Set(Array.from({ length: Math.max(0, b - a) }, (_, k) => a + k));
const slice = (a: number, b: number) => (b > a ? (b - 1 > a ? `${a}..${b - 1}` : `${a}`) : '');

function FormatLens() {
  const [which, setWhich] = useState<'book' | 'ex1'>('book');
  const g = which === 'book' ? BOOK : EX1;
  const csr = useMemo(() => toCSR(g), [g]);
  const csc = useMemo(() => toCSC(g), [g]);
  const coo = useMemo(() => toCOO(g), [g]);
  const st = storage(g);
  const [pick, setPick] = useState<Pick>({ kind: 'vertex', v: 3 });
  const none = new Set<number>();
  let csrHot = none, csrOn = none, cscHot = none, cscOn = none, cooOn = none, csrDstHot = none, cscSrcHot = none;
  let text: ReactNode = 'Click a vertex or an edge of the graph.';
  if (pick?.kind === 'vertex') {
    const v = pick.v;
    const [a, b] = [csr.srcPtrs[v], csr.srcPtrs[v + 1]];
    const [c, d] = [csc.dstPtrs[v], csc.dstPtrs[v + 1]];
    csrHot = new Set([v, v + 1]);
    csrOn = range(a, b);
    cscHot = new Set([v, v + 1]);
    cscOn = range(c, d);
    const outs = csr.dst.slice(a, b);
    const ins = csc.src.slice(c, d);
    text = (
      <>
        <b>Vertex {v}.</b> CSR: <code>srcPtrs[{v}] = {a}</code> and <code>srcPtrs[{v + 1}] = {b}</code>,{' '}
        {b > a ? (
          <>
            so its out-edges are <code>dst[{slice(a, b)}]</code> = {outs.join(', ')}: <span data-out>{outs.map((u) => `${v} → ${u}`).join(', ')}</span>.
          </>
        ) : (
          <>an empty slice: no edges leave {v}.</>
        )}{' '}
        CSC: <code>dstPtrs[{v}] = {c}</code> and <code>dstPtrs[{v + 1}] = {d}</code>,{' '}
        {d > c ? (
          <>
            so its in-edges are <code>src[{slice(c, d)}]</code> = {ins.join(', ')}: <span data-in>{ins.map((u) => `${u} → ${v}`).join(', ')}</span>.
          </>
        ) : (
          <>an empty slice: no edges arrive at {v}.</>
        )}{' '}
        COO has no per-vertex pointers: to find these edges there you would scan all {g.edges.length} entries.
      </>
    );
  } else if (pick?.kind === 'edge') {
    const e = pick.e;
    const [s, t] = g.edges[e];
    const k = csc.dstPtrs[t] + inNeighbours(csc, t).indexOf(s);
    cooOn = new Set([e]);
    csrDstHot = new Set([e]);
    cscSrcHot = new Set([k]);
    text = (
      <>
        <b>Edge {s} → {t}</b> is COO entry {e}: <code>src[{e}] = {s}</code>, <code>dst[{e}] = {t}</code>. Two reads, no
        search. In CSR the same edge is <code>dst[{e}]</code>, which you reach only through its source's row; in CSC it is{' '}
        <code>src[{k}]</code>, reached through its destination's column.
      </>
    );
  }
  const pickedVertex = pick?.kind === 'vertex' ? pick.v : -1;
  return (
    <Bench
      title="Format lens"
      task="Click a vertex: which slices light up? Then click an edge."
      foot={
        <>
          Amber cells are the array entries a thread would read: the darker pair are the pointers, the lighter run is
          the slice they bound. Small numbers are indices. Solid bold arrows are out-edges of the chosen vertex,
          dashed bold arrows its in-edges.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Graph">
          <button aria-pressed={which === 'book'} onClick={() => { setWhich('book'); setPick({ kind: 'vertex', v: 3 }); }}>the book's graph (Fig. 18.1)</button>
          <button aria-pressed={which === 'ex1'} onClick={() => { setWhich('ex1'); setPick({ kind: 'vertex', v: 0 }); }}>Exercise 1 graph (our reading of p. 450)</button>
        </span>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 5fr) minmax(0, 7fr)' }}>
        <div>
          <BfsGraph
            g={g}
            label={`The ${which === 'book' ? "book's" : 'Exercise 1'} graph; click a vertex or an edge`}
            look={(v) => (v === pickedVertex ? 'root' : 'plain')}
            edgeLook={(i, s, d) =>
              pick?.kind === 'edge' ? (i === pick.e ? 'pick' : 'dim') : s === pickedVertex ? 'out' : d === pickedVertex ? 'in' : 'dim'
            }
            onVertex={(v) => setPick({ kind: 'vertex', v })}
            onEdge={(e) => setPick({ kind: 'edge', e })}
          />
          <p className="muted" data-storage style={{ margin: '0.6rem 0 0' }}>
            Storage without the data array: dense matrix <b>{st.dense}</b>, CSR <b>{st.csr}</b> ({g.n + 1} + {g.edges.length}),
            CSC <b>{st.csc}</b>, COO <b>{st.coo}</b> ({g.edges.length} + {g.edges.length}).
          </p>
        </div>
        <div style={{ minWidth: 0 }}>
          <div className="narrate" role="status" style={{ marginBottom: '0.8rem' }}>{text}</div>
          <div style={{ overflowX: 'auto' }}>
            <div className="mx-label">CSR: outgoing edges</div>
            <ArrayRow id="csr-srcPtrs" name="srcPtrs" values={csr.srcPtrs} hot={csrHot} on={none} />
            <ArrayRow id="csr-dst" name="dst" values={csr.dst} hot={csrDstHot} on={csrOn} />
            <div className="mx-label" style={{ marginTop: '0.4rem' }}>CSC: incoming edges</div>
            <ArrayRow id="csc-dstPtrs" name="dstPtrs" values={csc.dstPtrs} hot={cscHot} on={none} />
            <ArrayRow id="csc-src" name="src" values={csc.src} hot={cscSrcHot} on={cscOn} />
            <div className="mx-label" style={{ marginTop: '0.4rem' }}>COO: one entry per edge</div>
            <ArrayRow id="coo-src" name="src" values={coo.src} hot={cooOn} on={none} />
            <ArrayRow id="coo-dst" name="dst" values={coo.dst} hot={cooOn} on={none} />
          </div>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- move the root
/** The explorer shows the answer, so it opens only once a prediction is committed. */
function RootGate() {
  const { progress } = useProgress();
  if (!progress.answers['ch18/start/p-root2']) {
    return <p className="muted ui">The explorer opens when you have committed a prediction.</p>;
  }
  return <RootExplorer />;
}

function RootExplorer() {
  const [root, setRoot] = useState(2);
  const [target, setTarget] = useState(4);
  const g = BOOK;
  const lv = useMemo(() => levels(g, root), [g, root]);
  const lv0 = useMemo(() => levels(g, 0), [g]);
  const csc = useMemo(() => toCSC(g), [g]);
  const tree = useMemo(() => new Set(treeEdges(g, root).map(([s, d]) => s + ',' + d)), [g, root]);
  const path = useMemo(() => backtrace(g, root, target), [g, root, target]);
  const onPath = new Set((path ?? []).slice(1).map((v, i) => path![i] + ',' + v));
  const changed = lv.filter((L, v) => L !== lv0[v]).length;
  // The walk back, step by step, reading each vertex's in-edges (CSC).
  const steps = path
    ? path.slice(1).reverse().map((v) => {
        const ins = inNeighbours(csc, v);
        const next = path[path.indexOf(v) - 1];
        return { v, ins, next };
      })
    : [];
  return (
    <Bench
      title="Move the root"
      task="Click a vertex to make it the root. Pick a vertex to walk back from."
      foot="Bold arrows are the BFS tree; amber arrows the path recovered backwards. Each vertex shows its level."
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Walk back from
          <select value={target} onChange={(e) => setTarget(+e.target.value)} aria-label="Walk back from">
            {Array.from({ length: g.n }, (_, v) => (
              <option key={v} value={v}>vertex {v}</option>
            ))}
          </select>
        </label>
        <span className="muted" data-root>root {root}</span>
        <button onClick={() => setRoot(0)} disabled={root === 0}>Root 0, as in Fig. 18.4(a)</button>
        <button onClick={() => setRoot(2)} disabled={root === 2}>Root 2, as in Fig. 18.4(b)</button>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 5fr) minmax(0, 6fr)' }}>
        <div>
          <BfsGraph
            g={g}
            label={`The book's graph with root ${root}`}
            look={(v) => (v === root ? 'root' : lv[v] === UNVISITED ? 'unvisited' : 'visited')}
            tag={(v) => showLevel(lv[v])}
            ring={(v) => (v === target ? 'pick' : null)}
            edgeLook={(_, s, d) => (onPath.has(s + ',' + d) ? 'path' : tree.has(s + ',' + d) ? 'tree' : 'dim')}
            onVertex={setRoot}
          />
        </div>
        <div style={{ minWidth: 0 }}>
          <table className="threads" aria-label="Levels by root">
            <thead>
              <tr>
                <th>vertex</th>
                {Array.from({ length: g.n }, (_, v) => <th key={v}>{v}</th>)}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>root 0</td>
                {lv0.map((L, v) => <td key={v}>{showLevel(L)}</td>)}
              </tr>
              <tr>
                <td>root {root}</td>
                {lv.map((L, v) => (
                  <td key={v} data-level={v} style={L !== lv0[v] ? { fontWeight: 650 } : undefined}>{showLevel(L)}</td>
                ))}
              </tr>
            </tbody>
          </table>
          <p data-changed={changed} style={{ margin: '0.6rem 0' }}>
            {root === 0 ? 'This is root 0.' : <><b>{changed} of {g.n}</b> vertices have a different level than with root 0.</>}
          </p>
          <div className="narrate" data-path={path ? path.join(' → ') : 'none'}>
            {path ? (
              path.length === 1 ? (
                <>Vertex {target} is the root: the path is just {target}.</>
              ) : (
                <>
                  {steps.map((s) => (
                    <div key={s.v}>
                      {s.v} (level {lv[s.v]}): in-edges from {s.ins.map((u) => `${u} (level ${showLevel(lv[u])})`).join(', ')}; step to {s.next}.
                    </div>
                  ))}
                  <div style={{ marginTop: '0.3rem' }}>
                    <b>Path: {path.join(' → ')}</b>, {path.length - 1} edges.
                  </div>
                </>
              )
            ) : (
              <>Vertex {target} cannot be reached from {root}: it has no level and no path.</>
            )}
          </div>
        </div>
      </div>
    </Bench>
  );
}

export function S180() {
  return (
    <>
      <p>
        The book uses breadth-first search, BFS, as its example of work whose amount and place change while the
        program runs: which vertices need attention at the next step depends on what the current step found. BFS
        itself is not part of an LLM inference stack, and this course treats it as a vehicle. It carries three
        techniques that you will use:
      </p>
      <ul>
        <li>
          <b>launch threads only for what changed</b>: a frontier, the list of vertices the last step discovered
          (§18.5);
        </li>
        <li>
          <b>build that list in a queue private to each block</b> and commit it with one global atomic per block,
          which is chapter 12's privatised filter plus a way out when the private queue fills up (§18.6);
        </li>
        <li>
          <b>run many dependent steps inside one launch</b>, separated by a barrier across the whole grid (§18.7).
        </li>
      </ul>
      <Source pages="425, 449" />
      <p>
        Token-by-token decoding has the shape the book describes for small frontiers (our reading): every step
        launches many short kernels, and each launch has a fixed cost that the little work inside it cannot hide. The
        book's remedy is one launch for many levels. Inference engines reach for the same remedy in other forms,
        recording a step's launches once and replaying them.
      </p>
      <Beyond title="Bridge (ours)">
        <p>
          The book discusses launch overhead only for BFS on graphs with small frontiers, such as road networks (p.
          445). The mapping to decoding is ours.
        </p>
      </Beyond>
      <Today checked="2026-11-20 (notes/today/ch18-facts.md D3, D5, D6)">
        <p>
          llama.cpp turns CUDA graphs on by default when you build llama.cpp itself, and HIP graphs on by default.
          Metal has no graph API, so its backend encodes with concurrent dispatches and memory barriers and spreads the
          encoding over CPU threads. PyTorch's CUDA graphs prohibit operations that synchronise with the CPU, such as{' '}
          <code>.item()</code>: exactly the per-level read of the frontier size a BFS host loop makes. MLX's
          documentation calls work on small arrays "overhead bound".
        </p>
      </Today>

      <h2>Graphs in chapter 17's formats</h2>
      <p>
        A graph is a set of vertices joined by edges. The chapter's edges are directed: an edge goes from a source
        vertex to a destination vertex, and a two-way relation is two edges. Number the vertices from 0 and the graph is
        a square matrix with a 1 in row <i>i</i>, column <i>j</i> for each edge from <i>i</i> to <i>j</i>. A row holds a
        vertex's outgoing edges, a column its incoming ones. Almost every entry is 0: the running example has 9
        vertices and 15 edges, where a fully connected graph of 9 vertices would have 72.
      </p>
      <Source section="18.1" pages="425–427" figs="Figs. 18.1, 18.2" />
      <p>
        So the sparse formats of <a href={href('ch17', '17.2')}>chapter 17</a> apply, under new names. CSR keeps the rows as{' '}
        <code>srcPtrs</code> (row offsets) and <code>dst</code> (column indices). CSC keeps the columns as{' '}
        <code>dstPtrs</code> and <code>src</code>. COO keeps each edge as a <code>src</code>, <code>dst</code> pair.
        Every stored value would be 1, so the data array is dropped: an index existing means an edge exists. That
        leaves 25 numbers for CSR, 10 offsets and 15 destinations, against 81 for the dense matrix.
      </p>
      <Source section="18.1" pages="427–428" figs="Fig. 18.3" />
      <p>
        Read it the way the book does: <code>srcPtrs[3] = 7</code> and <code>srcPtrs[4] = 9</code>, so vertex 3's
        outgoing edges are entries 7 and 8 of <code>dst</code>, which hold 4 and 8. The edges are 3 → 4 and 3 → 8.
        Check that in the lens, then click a vertex that only receives edges, and an edge.
      </p>
      <FormatLens />
      <p>
        Each format makes one question cheap. CSR answers "where can I go from <i>v</i>?" with one slice. CSC answers
        "who points at <i>v</i>?" with one slice. COO answers "where does edge <i>e</i> start and end?" with two reads.
        Ask the wrong format and you search: CSR for the vertices pointing at 6 means looking through every row. The
        next page builds a BFS kernel on each format, and each kernel asks the question its format answers cheaply.
      </p>
      <Source section="18.1" pages="428–429" />
      <p>
        Which kernel suits a graph also depends on how its edges are spread. A vertex's degree is its number of edges.
        Road networks have low, even degrees: an intersection joins a handful of roads. Follower graphs are broad, with
        a few popular accounts holding orders of magnitude more edges than the rest. The next page runs both kinds.
      </p>
      <Source section="18.1" pages="428" />
      <Ask q={questionsS180['ch18/start/q-csr-read']} label="Read the arrays" />
      <Ask q={questionsS180['ch18/start/q-csc-in']} label="Read the arrays" />

      <h2>Breadth-first search</h2>
      <p>
        BFS finds the smallest number of edges from a root to every other vertex, and labels each vertex with that
        number, its level. The root is level 0, its out-neighbours level 1, their unvisited out-neighbours level 2, and
        so on. With root 0, the example has 1 and 2 at level 1, 3 to 7 at level 2, and 8 at level 3.
      </p>
      <Source section="18.2" pages="429–430" figs="Fig. 18.4(a)" />
      <p>
        Before you move the root, predict what one edge's move does to a single vertex. Then the explorer opens, and you
        can move the root anywhere.
      </p>
      <Simplification>
        <p>
          The explorer's levels come from an ordinary sequential BFS. Where a vertex has several in-neighbours one level
          lower, the walk back takes the lowest-numbered: that rule is ours, and any choice gives a shortest path.
        </p>
      </Simplification>
      <Predict
        id="ch18/start/p-root2"
        prompt={
          <>
            Move the root one edge along, from 0 to 2 (0 → 2 is an edge). At what level is vertex 4 now? It was at level
            2.
          </>
        }
      >
        {(v) => (
          <>
            <p>
              {v === 4 ? 'Right: ' : 'It is at '}level 4, by 2 → 7 → 0 → 1 → 4. From 2 the search must go the long way
              round, back through 0, to reach 1 and then 4. Now click any vertex in the explorer below to make it
              the root.
            </p>
          </>
        )}
      </Predict>
      <RootGate />
      <p>
        The edges a search uses to reach each vertex form the BFS tree, drawn bold: one edge into every reached vertex
        but the root. To recover a shortest path to a vertex, start there and step to any in-neighbour exactly one level
        lower, until you reach the root. With root 0, from vertex 8: its in-neighbours 3, 4 and 6 are all at level 2,
        so any of them will do; the explorer takes 3, then 1, then 0. Notice what the walk reads: each vertex's
        incoming edges. That is the CSC question, not the CSR one.
      </p>
      <Source section="18.2" pages="430" />
      <Ask q={questionsS180['ch18/start/q-path']} />
      <p>
        Chip design uses this directly. In maze routing, the wiring grid is the graph: each free block is a vertex,
        each step a wire can take to a neighbouring block is an edge, and blocks already used are blockages to route
        around. A search from one terminal spreads as a wavefront until it reaches the other, and the walk back is the
        shortest wire. The next page plays that wavefront as an animation.
      </p>
      <Source section="18.2" pages="430–431" figs="Fig. 18.5" />

      <h2>The chapter in this course</h2>
      <p>This chapter is condensed. Here is where the book's sections went.</p>
      <table className="data">
        <thead>
          <tr>
            <th>Book</th>
            <th>Here</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>§18.1, §18.2 (pp. 425–431)</td>
            <td>this page</td>
          </tr>
          <tr>
            <td>§18.3 to §18.5 (pp. 431–442)</td>
            <td>
              <a href={href('ch18', '18.3')}>18.3 Vertices, edges or a frontier</a>: push and pull, edge-centric, work
              efficiency, frontiers, compare-and-swap
            </td>
          </tr>
          <tr>
            <td>§18.6 to §18.8 (pp. 442–449)</td>
            <td>
              <a href={href('ch18', '18.6')}>18.6 Private queues and one launch for every level</a>
            </td>
          </tr>
          <tr>
            <td>ours</td>
            <td>
              <a href={href('ch18', 'labs')}>Lab 18.1</a>, optional: frontier BFS with a privatised queue
            </td>
          </tr>
          <tr>
            <td>§18.9, §18.10 (pp. 449–450)</td>
            <td>
              <a href={href('ch18', 'wrap')}>Summary and exercises</a>
            </td>
          </tr>
        </tbody>
      </table>
      <p>
        Cut to a sentence or left out: the heuristics routers use to choose between equally short wires (p. 431),
        small-world graphs (p. 436), the full linear-algebra formulation of BFS (p. 438), and the open problems of
        graphs larger than GPU memory and of reordering vertices (p. 449).
      </p>
      <Aside>
        <p>
          Chapter 17 is condensed too, and covers the formats this page borrows on its page <a href={href('ch17', '17.2')}>17.2</a>.
        </p>
      </Aside>

      <h2>Before you start</h2>
      <p>
        Six questions on what this chapter assumes from chapters 4, 9, 11, 12 and 17. Each comes back in{' '}
        <a href={href('review')}>review</a>.
      </p>
      <Ask q={questionsS180['ch18/start/q-divergence']} label="From chapter 4" />
      <Ask q={questionsS180['ch18/start/q-resident']} label="From chapters 4 and 11" />
      <Ask q={questionsS180['ch18/start/q-workeff']} label="From chapter 11" />
      <Ask q={questionsS180['ch18/start/q-slot']} label="From chapter 12" />
      <Ask q={questionsS180['ch18/start/q-scope']} label="From chapter 9" />
      <Ask q={questionsS180['ch18/start/q-rowthread']} label="From chapters 17 and 6" />
    </>
  );
}
