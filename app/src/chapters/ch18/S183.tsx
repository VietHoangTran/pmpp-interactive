import { useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { changedLines, region } from '../../sim/bfsvisit';
import { functionSource } from '../../sim/toolchain';
import { BFS_TIERS } from './BfsGraph';
import { LevelExplorer, type ExplorerRequest } from './LevelExplorer';
import { VisitStepper } from './VisitStepper';
import { WavefrontScene } from './WavefrontScene';

// Section 18.3 of this course: the book's §18.3 to §18.5 (pp. 431-442). What the book says
// comes from notes/ch18.md; vendor facts from notes/today/ch18-facts.md (A6, A9, A12, B4, C4,
// C5, D1 and "Already settled"). Models: src/sim/bfs.ts (the explorer, the scene) and
// src/sim/bfsvisit.ts (the stepper). Listings: labs/ch18/snippets/s183_variants.cu.

export const questionsS183: Record<string, Question> = {
  'ch18/18.3/q-who-loops': {
    kind: 'choice',
    id: 'ch18/18.3/q-who-loops',
    prompt: (
      <>
        The book's graph, root 0, pull kernel, the call that labels level 2. Which thread scans its whole list of
        incoming edges and labels nothing, and why does no other?
      </>
    ),
    options: [
      {
        label: "Vertex 8's: its in-neighbours 3, 4 and 6 are all still unvisited, so it has no parent at level 1. Every other unvisited vertex finds one and breaks.",
        correct: true,
      },
      {
        label: "Vertex 0's: it has no in-edge from level 1.",
        why: 'Vertex 0 is the root, at level 0. Its thread fails the unvisited check on line 05 and never loops.',
      },
      {
        label: "Vertex 4's: it has two in-edges and must check both.",
        why: 'Its first in-neighbour in CSC order is 1, at level 1. It labels itself and breaks after one.',
      },
      {
        label: 'None: every thread that loops labels its vertex.',
        why: 'A thread loops whenever its vertex is unvisited, whether or not a parent exists yet.',
      },
    ],
    explain: (
      <p>
        In Fig. 18.9 the threads on 3 to 7 each find a parent at level 1 (1 or 2) and stop. Vertex 8 is two levels away
        from the frontier, so its thread reads all three in-edges for nothing. That is pull's waste early in a search:
        it loops for every unvisited vertex, near the frontier or not.
      </p>
    ),
    source: 'ch. 18 §18.3, pp. 434–435, Fig. 18.9',
  },
  'ch18/18.3/q-push-pull': {
    kind: 'choice',
    id: 'ch18/18.3/q-push-pull',
    prompt: (
      <>
        A social graph at level 3 of 6. The last level holds 40% of the vertices and 10% are still unvisited. Push or
        pull for this level?
      </>
    ),
    options: [
      {
        label: 'Pull: only the 10% loop, and with so many parents in the last level most break at their first in-edge.',
        correct: true,
      },
      {
        label: 'Push: the last level is where the work starts, so start from it.',
        why: 'Push would walk the long out-lists of 40% of the graph, mostly to find neighbours already labelled.',
      },
      {
        label: 'Either: both read every edge once over the whole search.',
        why: 'Push reads each edge once in total; pull re-reads edges level after level. Per level, the costs are very different.',
      },
    ],
    explain: (
      <p>
        Push loops for the last level's vertices and walks every edge they have. Pull loops for the unvisited and stops
        at the first parent. Late in a search on a graph with high, uneven degrees, the unvisited are few and their
        parents plentiful, which is the case direction-optimised BFS switches for.
      </p>
    ),
    source: 'ch. 18 §18.3, pp. 435–436',
  },
  'ch18/18.3/q-ex2': {
    kind: 'choice',
    id: 'ch18/18.3/q-ex2',
    prompt: (
      <>
        Exercise 2: you write the host loop of a direction-optimised BFS. What must be on the device, and what does the
        host need at each level to choose the kernel?
      </>
    ),
    options: [
      {
        label: 'CSR for push and CSC for pull (one pair if the graph is symmetric), and a measure of the frontier against the unvisited part, such as edges out of each.',
        correct: true,
      },
      {
        label: 'CSR only, and the level number: switch at a fixed level.',
        why: 'Pull reads incoming edges, which CSR does not give. And the right level depends on the graph, which a fixed number ignores.',
      },
      {
        label: 'COO only, since it serves both directions.',
        why: 'COO serves the edge-centric kernel. Push needs a vertex\'s out-edges as a slice, pull its in-edges.',
      },
    ],
    explain: (
      <p>
        The book says only that the switch comes earlier for graphs with high, uneven degrees. cuGraph's rule, after
        Beamer 2012, compares edges out of the frontier with edges out of unvisited vertices: m_f and m_u in the
        explorer's table. The host has to bring those numbers back each level, one more read on top of the flag or
        frontier size.
      </p>
    ),
    source: 'ch. 18 §18.3, p. 436; Exercise 2, p. 450; cuGraph: notes/today/ch18-facts.md D1',
  },
  'ch18/18.3/q-hub-checks': {
    kind: 'numeric',
    id: 'ch18/18.3/q-hub-checks',
    prompt: (
      <>
        A vertex with 1,000 out-edges is never in the last level during a search whose host loop makes 10 kernel
        calls, the last one finding nothing. How many times, in total, do edge-centric threads read its level?
      </>
    ),
    answer: 10000,
    unit: 'reads',
    explain: (
      <p>
        Each of its 1,000 edges has a thread in every call, and each thread reads the source's level to find out it has
        nothing to do: 1,000 edges × 10 calls. A push thread reads it once per call, 10 in all, and skips the whole
        list. Ten calls is a search of 9 levels after the root, since the last call finds nothing.
      </p>
    ),
    source: 'ch. 18 §18.4, p. 438',
  },
  'ch18/18.3/q-edge-gain': {
    kind: 'choice',
    id: 'ch18/18.3/q-edge-gain',
    prompt: <>When does edge-centric BFS beat vertex-centric push?</>,
    options: [
      {
        label: 'On a small graph with high degree variance: more threads to fill the device, and no per-thread loop for a long list to stretch.',
        correct: true,
      },
      {
        label: 'On a large road network: it reads every edge once.',
        why: 'It reads every edge at every level, and a road network has many levels.',
      },
      {
        label: 'Whenever memory is short: COO is the smallest format.',
        why: 'COO stores two numbers per edge; CSR and CSC store one per edge plus one per vertex.',
      },
    ],
    explain: (
      <p>
        One thread per edge exposes m threads instead of n, and every thread does the same constant work, so a hub
        cannot hold up its warp. The price is checking all m edges at every level and the larger COO arrays.
      </p>
    ),
    source: 'ch. 18 §18.4, pp. 437–438',
  },
  'ch18/18.3/q-chain': {
    kind: 'choice',
    id: 'ch18/18.3/q-chain',
    prompt: (
      <>
        A chain of n vertices, searched from one end, so d = n − 1. How many threads does vertex-centric push launch in
        total, against the frontier version?
      </>
    ),
    options: [
      { label: 'About n² against about n: O(d·n) against O(n + m).', correct: true },
      { label: 'About n each: both visit every vertex once.', why: 'Push launches a thread for every vertex at every level, visited or not.' },
      { label: 'About n log n against n.', why: 'Nothing halves here: the chain has n − 1 levels, one vertex each.' },
    ],
    explain: (
      <p>
        n threads per call for n calls is n². The frontier version launches one thread for the single vertex of each
        level. Road-like graphs, with many levels of few vertices, are where vertex-centric BFS is at its least
        work-efficient. Try the chain in the explorer.
      </p>
    ),
    source: 'ch. 18 §18.5, pp. 438–439',
  },
  'ch18/18.3/q-why-cas': {
    kind: 'choice',
    id: 'ch18/18.3/q-why-cas',
    prompt: (
      <>
        Why does Fig. 18.6 get away, in practice, with an unsynchronised check-then-label, while Fig. 18.12 must use
        compare-and-swap?
      </>
    ),
    options: [
      {
        label: 'In Fig. 18.6 a duplicate writes the same value. In Fig. 18.12 each winner also appends, so a duplicate lands twice in the frontier and is expanded twice.',
        correct: true,
      },
      {
        label: 'Fig. 18.12 runs fewer threads, so collisions are likelier.',
        why: 'Fewer threads make collisions rarer, not likelier. The difference is what a collision does.',
      },
      {
        label: 'Fig. 18.6 is correct by the memory model; Fig. 18.12 is not.',
        why: 'Fig. 18.6 is a data race too, undefined by the memory model. It merely works in practice.',
      },
    ],
    explain: (
      <p>
        Without frontiers a duplicate is idempotent. With frontiers each check-then-label also claims a slot, and
        claiming twice is not idempotent. Fig. 18.6 is still formally a data race, which is why the corrected listing
        uses relaxed atomics.
      </p>
    ),
    source: 'ch. 18 §18.5, pp. 440–441',
  },
  'ch18/18.3/q-stale': {
    kind: 'choice',
    id: 'ch18/18.3/q-stale',
    prompt: (
      <>
        The code declares <code>unsigned int e = UNVISITED;</code> once, above the edge loop. The thread on vertex 2
        meets neighbour 3, already won by another thread this level, and then 4, unvisited. What happens?
      </>
    ),
    options: [
      {
        label: "The failed compare-and-swap on 3 writes 3's level into e. The one on 4 then compares UNVISITED with that level and fails, so 4 is missed.",
        correct: true,
      },
      {
        label: 'Both work: compare-and-swap resets e when it fails.',
        why: 'It does the opposite: on failure it loads the value it found into e.',
      },
      {
        label: "4 is labelled twice: e still holds UNVISITED from the declaration.",
        why: 'e changed when the swap on 3 failed. Step it in the stepper to see.',
      },
    ],
    explain: (
      <p>
        After a failed compare-and-swap the expected variable holds the current value (here, level 2). Reusing it
        compares 4's UNVISITED with 2: not equal, so the swap fails again and loads UNVISITED back into e, too late.
        Vertex 4 is labelled at a later level, or never: in this graph only 2 points to 4. Fig. 18.14 avoids this by
        declaring its variable fresh on every call.
      </p>
    ),
    source: 'ch. 18 §18.5, p. 441; C++ semantics: notes/today/ch18-facts.md A6',
  },
  'ch18/18.3/q-weak-loop': {
    kind: 'choice',
    id: 'ch18/18.3/q-weak-loop',
    prompt: (
      <>
        Metal has only a weak compare-and-swap, which may fail even when the values are equal. Which loop retries
        exactly the spurious failures?
      </>
    ),
    options: [
      {
        label: 'while (!cas(e) && e == UNVISITED) {}',
        correct: true,
      },
      {
        label: 'while (!cas(e)) {}',
        why: 'After a real failure e holds the winner\'s level, so the retry compares that level with itself, succeeds, and appends a duplicate, or overwrites an older level with the current one.',
      },
      {
        label: 'while (!cas(e)) e = UNVISITED;',
        why: 'Once another thread has won, level[v] is never UNVISITED again, so this never ends.',
      },
    ],
    explain: (
      <p>
        A spurious failure leaves <code>e</code> at UNVISITED; a real one loads the winner's level into it. Testing{' '}
        <code>e</code> tells the two apart: retry on the first, give up on the second, and append only if the swap
        succeeded. Try all three in the stepper with one spurious failure allowed.
      </p>
    ),
    source: 'notes/today/ch18-facts.md A6, B4, C4 (MSL spec §6.16.4.4)',
  },
};

// ---------------------------------------------------------------- the five listings
const FIGS = [
  { id: 'fig18_6', label: 'Fig. 18.6 push' },
  { id: 'fig18_8', label: 'Fig. 18.8 pull' },
  { id: 'fig18_10', label: 'Fig. 18.10 edge' },
  { id: 'fig18_12', label: 'Fig. 18.12 frontier' },
  { id: 'fig18_14', label: 'Fig. 18.14 visit' },
];
const WHAT_CHANGED: Record<string, string> = {
  fig18_6: 'Lines 05 and 09 to 11: level[] and the flag through relaxed atomic loads and stores.',
  fig18_8: 'Lines 05 and 09 to 11: the same, for the pull kernel.',
  fig18_10: 'Lines 06 and 08 to 10: the same, for the edge-centric kernel.',
  fig18_12: 'No line changes: the frontier kernel touches level[] only through Fig. 18.14.',
  fig18_14: 'A HIP branch: the legacy atomicCAS, which is strong on HIP. The CUDA lines are the book\'s.',
};

function Listings() {
  const src = useLabFile('ch18/snippets/s183_variants.cu');
  const [fixed, setFixed] = useState(false);
  const [which, setWhich] = useState(0);
  if (!src) return null;
  const tabs = FIGS.map((f) => {
    const printed = region(src, f.id + '_printed') ?? '// region not found';
    const corrected = region(src, f.id + '_fixed') ?? '// region not found';
    return {
      label: f.label,
      code: fixed ? corrected : printed,
      tiers: BFS_TIERS,
      highlight: fixed ? changedLines(printed, corrected) : [],
      status: 'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware',
    };
  });
  return (
    <div className="wide" style={{ maxWidth: '58rem' }}>
      <div className="row ui" style={{ marginBottom: '-0.4rem' }}>
        <span className="seg" role="group" aria-label="Listing version">
          <button aria-pressed={!fixed} onClick={() => setFixed(false)}>as printed</button>
          <button aria-pressed={fixed} onClick={() => setFixed(true)}>corrected</button>
        </span>
        <span className="seg" role="group" aria-label="Figure">
          {FIGS.map((f, i) => (
            <button key={f.id} aria-pressed={which === i} onClick={() => setWhich(i)}>{f.label.split(' ')[1]}</button>
          ))}
        </span>
      </div>
      <Code
        key={which}
        tabs={[tabs[which]]}
        caption={
          <>
            {FIGS[which].label}, {fixed ? 'corrected' : 'as printed'}, from <code>labs/ch18/snippets/s183_variants.cu</code>.{' '}
            {fixed ? <>Marked: what changed. {WHAT_CHANGED[FIGS[which].id]}</> : <>The book's text and line numbers. The file adds the graph structs and the <code>UNVISITED</code> the book never defines.</>}
          </>
        }
      />
    </div>
  );
}

/**
 * The Metal tab's status, from this machine's reports: lab 18.1's "provided frontier" test runs
 * ref_bfs_frontier, which calls ref_visit_vertex, in every variant.
 */
function metalVisitStatus(reports: Reports): string {
  const rs = (reports.metal?.results ?? []).filter((r) => r.lab === 'ch18/lab1_frontier' && r.built);
  const tests = rs.flatMap((r) => r.tests.filter((t) => t.case.startsWith('provided frontier')));
  if (!tests.length) return 'unverified (no run of lab 18.1 on this machine)';
  return tests.every((t) => t.pass)
    ? "verified on Metal (lab 18.1's provided frontier test, which calls it, passes on this machine)"
    : 'unverified (failed on this machine)';
}

/** Code inside the backends table: allowed to wrap, so no column is squeezed by one long name. */
const TC = ({ children }: { children: ReactNode }) => <code style={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>{children}</code>;

/** Fig. 18.14 as the course writes it for CUDA and HIP, and the Metal form lab 18.1 provides. */
function VisitOnBackends() {
  const cu = useLabFile('ch18/snippets/s183_variants.cu');
  const metal = useLabFile('ch18/lab1_frontier/provided/ref_kernels.metal');
  const { reports } = useReports();
  if (!cu || !metal) return null;
  const fig = region(cu, 'fig18_14_fixed') ?? '// region not found';
  const fn = functionSource(metal, 'ref_visit_vertex');
  return (
    <Code
      tabs={[
        { label: 'CUDA and HIP', code: fig, tiers: BFS_TIERS, highlight: changedLines(region(cu, 'fig18_14_printed') ?? '', fig), status: 'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware' },
        { label: 'Metal', tiers: BFS_TIERS, code: fn ? fn.text : '// ref_visit_vertex not found in labs/ch18/lab1_frontier/provided/ref_kernels.metal', status: metalVisitStatus(reports) },
      ]}
      caption={
        <>
          The CUDA tab is the corrected Fig. 18.14 from <code>labs/ch18/snippets/s183_variants.cu</code>, its HIP branch
          marked. The Metal tab is the visit function lab 18.1 provides, from{' '}
          <code>labs/ch18/lab1_frontier/provided/ref_kernels.metal</code>: the retry loop, written as an early return.
        </>
      }
    />
  );
}

export function S183() {
  const [request, setRequest] = useState<ExplorerRequest | null>(null);
  const show = (variant: 'pull' | 'edge', step: number) => setRequest({ preset: 1, root: 0, variant, step, key: Date.now() });
  return (
    <>
      <p>
        This page runs one level of BFS four ways on four small graphs, counts the work each way does, and ends with
        the step every later kernel depends on: making a vertex join the next frontier exactly once, with
        compare-and-swap, on CUDA, HIP and Metal.
      </p>

      <h2>One launch per level</h2>
      <p>
        Every kernel on this page labels one level per launch. The root starts at level 0 and every other vertex at{' '}
        <code>UNVISITED</code>. The kernel for level L labels the unvisited neighbours of the level L − 1 vertices, and
        the host launches it again for L + 1 until a launch labels nothing. The kernel boundary is doing real work: it
        is the only thing that guarantees every vertex of level L − 1 is labelled before any thread looks for level L.
        Section 18.7 replaces it with a barrier inside one launch.
      </p>
      <Source section="18.3" pages="432" />
      <p>
        The section's five kernels are below, as printed and as this course corrects them. Read Fig. 18.6 first; the
        others come up in turn.
      </p>
      <Aside>
        <p>
          The book never defines its graph structs or <code>UNVISITED</code>; p. 433 suggests{' '}
          <code>cuda::std::numeric_limits&lt;unsigned&gt;</code>, which names the trait without its{' '}
          <code>max()</code>. The file uses <code>UINT_MAX</code>.
        </p>
      </Aside>
      <Listings />

      <h2>Push: threads on the last level</h2>
      <p>
        Fig. 18.6 gives every vertex a thread. A thread whose vertex is at level L − 1 walks its out-edges in CSR (lines
        06 to 08) and labels every neighbour still <code>UNVISITED</code> (lines 09 and 10). It also sets{' '}
        <code>newVertexVisited</code>, the flag the host reads to decide whether to launch again. The threads sit on
        parents and look for children, so the book calls this top-down, or push. With root 0, in the call that labels
        level 2 only the threads on 1 and 2 get past line 05: that is Fig. 18.7.
      </p>
      <Source section="18.3" pages="432–433" figs="Figs. 18.6, 18.7" />
      <p>
        Two threads can reach the same unvisited neighbour, both see <code>UNVISITED</code>, and both write the same
        level. The book calls this benign: the write is idempotent, so in practice nothing goes wrong. It adds that the
        code violates the C++ memory model and is not guaranteed to work, and that a careful programmer uses atomics.
        The corrected listing does: relaxed atomic loads and stores of <code>level[]</code> and the flag, which ask for
        no ordering and remove the race.
      </p>
      <Source section="18.3" pages="433–434" />
      <Today checked="2026-11-20 (notes/today/ch18-facts.md, “Already settled”: ch10 A1, ch14 A9)">
        <p>
          CUDA's memory model makes a data race undefined behaviour whether or not the racing writes store the same
          value. HIP documents no rule of its own.
        </p>
      </Today>

      <h2>Pull: threads on the unvisited</h2>
      <p>
        Fig. 18.8 turns it round. A thread whose vertex is still <code>UNVISITED</code> scans its in-edges in CSC and
        looks for a neighbour at level L − 1. One is enough: it labels its own vertex and breaks (line 12). Each thread
        writes only its own level, so the flag is the only shared write. The threads sit on children looking for a
        parent: bottom-up, or pull.
      </p>
      <Source section="18.3" pages="434–435" figs="Figs. 18.8, 18.9" />
      <Ask q={questionsS183['ch18/18.3/q-who-loops']} />
      <p>
        Which is better depends on the level, for two reasons. Push walks a whole out-list; pull can stop at its first
        parent, which matters little on a road network and a lot on a social graph with long, uneven lists. And push
        loops for the vertices of the last level, pull for every vertex not yet visited. Early in a search the last
        level is small and most vertices are unvisited, so push wins. Late, the last level is large and few vertices
        are left, each likely to find a parent at once, so pull wins.
      </p>
      <Source section="18.3" pages="435–436" />
      <p>
        Direction-optimised BFS pushes first and switches to pull. It needs out-edges for the push levels and in-edges
        for the pull levels: CSR and CSC both, unless the graph is symmetric (every edge has its reverse), when one pair
        of arrays serves both. On when to switch, the book says only that graphs with high, uneven degrees switch
        earlier.
      </p>
      <Source section="18.3" pages="436" />
      <Today checked="2026-11-20 (notes/today/ch18-facts.md D1)">
        <p>
          cuGraph's BFS is direction-optimising, after Beamer 2012, and requires a symmetric graph. It goes bottom-up
          when the edges out of the frontier (m_f), scaled by a tuned factor, outnumber the edges out of unvisited
          vertices (m_u). The explorer below shows both counts at every call.
        </p>
      </Today>
      <Ask q={questionsS183['ch18/18.3/q-push-pull']} />
      <Ask q={questionsS183['ch18/18.3/q-ex2']} label="Exercise 2, p. 450" />

      <h2>Edge-centric: a thread per edge</h2>
      <p>
        Fig. 18.10 gives every edge a thread. It reads its edge's source from COO; if the source is at level L − 1 and
        the destination is still <code>UNVISITED</code>, it labels the destination. There is no loop, so no thread is
        stretched by a long list: every thread does the same work, which is chapter 6's idea of re-mapping threads to
        work. There are also more threads, which helps a graph too small to fill the device one thread per vertex. The
        costs: every edge is checked at every level, and COO stores two numbers per edge.
      </p>
      <Source section="18.4" pages="436–438" figs="Figs. 18.10, 18.11" />
      <p>
        The book's example of the first cost is a vertex with many edges that is irrelevant at some level. Edge-centric
        launches a thread per edge, and each one finds out the same thing; push finds out once and skips the list.
      </p>
      <Source section="18.4" pages="438" />
      <Ask q={questionsS183['ch18/18.3/q-hub-checks']} label="Work it out" />
      <Ask q={questionsS183['ch18/18.3/q-edge-gain']} />
      <p>
        These kernels look like chapter 17's SpMV kernels for a reason: a BFS level can be written as a sparse
        matrix–vector product, the view behind the GraphBLAS API. The book gives it a paragraph; this course leaves it
        there.
      </p>
      <Source section="18.4" pages="438" />
      <Today checked="2026-11-20 (notes/today/ch18-facts.md A12)">
        <p>GraphBLAS C API 2.1.0; its reference implementation runs on the CPU, its CUDA folder a draft.</p>
      </Today>

      <h2>Run them side by side</h2>
      <p>
        Exercise 1 asks, for every call on its 8-vertex graph, how many threads each kernel launches, how many loop and
        how many label. Predict two of those counts; each reveal opens the explorer on that call.
      </p>
      <Aside>
        <p>
          The answers are ours: the book prints none, and the edge directions are our reading of the drawing on p.
          450. The model's unit tests check them against the notes.
        </p>
      </Aside>
      <Predict
        id="ch18/18.3/p-ex1c-pull"
        prompt={<>Exercise 1 graph, root 0, pull kernel, second call (it labels level 2). How many threads loop over their incoming edges?</>}
        unit="threads"
      >
        {(v) => (
          <>
            <p>
              {v === 5 ? 'Right: ' : v === 3 ? 'That counts the threads that label. The answer is ' : 'It is '}
              5: the unvisited vertices 1, 3, 4, 6 and 7. Of those, 3 find a parent at level 1 and label their vertex
              (1, 3 and 7); 4 and 6 have none yet.
            </p>
            <button onClick={() => show('pull', 2)}>Show it in the explorer</button>
          </>
        )}
      </Predict>
      <Predict
        id="ch18/18.3/p-ex1c-edge"
        prompt={<>Same graph, edge-centric kernel, third call. How many threads find their edge's source at level 2?</>}
        unit="threads"
      >
        {(v) => (
          <>
            <p>
              {v === 8 ? 'Right: ' : v === 4 ? 'That counts the threads that may label. The answer is ' : 'It is '}
              8: the edges out of 1, 3 and 7. Four of them may label: 1 → 4, 7 → 4, 3 → 6 and 7 → 6, two writers for each
              of 4 and 6, the benign duplicate again.
            </p>
            <button onClick={() => show('edge', 3)}>Show it in the explorer</button>
          </>
        )}
      </Predict>
      <p>
        Things to try. The book's graph, root 0, call 2, push and then pull: Figs. 18.7 and 18.9. The chain, push
        against frontier: watch the launched totals. The star from a leaf, push at call 2: the hub's thread loops alone
        while its warp waits, and edge-centric spreads the same edges one per thread. Direction-optimised with the
        switch slider, watching m_f and m_u.
      </p>
      <Simplification>
        <p>
          A step is one kernel call; nothing is timed. Blocks and warps are 4 threads wide, so a whole call fits on
          screen. Every thread of a call sees the levels as they were before the call, so where two threads may write the same vertex the labels written are a range. The frontier kernel
          appends in one possible order (threads in index order). Figs. 18.7 to 18.13 print each vertex's final level
          even at the level 1 to 2 step; the explorer shows levels as of the current call.
        </p>
      </Simplification>
      <LevelExplorer request={request} />

      <h2>Work efficiency</h2>
      <p>
        Every kernel so far checks every vertex or every edge at every level. That keeps them simple and free of
        synchronisation, and makes most of their threads useless. For a graph with n vertices, m edges and d levels
        after the root's, the book counts:
      </p>
      <Source section="18.5" pages="438–439" />
      <table className="data">
        <thead>
          <tr><th>Kernel</th><th>Work</th><th>Why</th></tr>
        </thead>
        <tbody>
          <tr><td>ideal</td><td>O(n + m)</td><td>each vertex and edge once</td></tr>
          <tr><td>vertex-centric push</td><td>O(d·n + m)</td><td>every vertex checked each level; each edge walked once, from its source's level</td></tr>
          <tr><td>vertex-centric pull</td><td>O(d·n + d·m)</td><td>every vertex each level; an edge can be re-read every level until its destination is labelled</td></tr>
          <tr><td>edge-centric</td><td>O(d·m)</td><td>every edge each level</td></tr>
          <tr><td>push with frontiers</td><td>O(n + m)</td><td>threads only for the last level's vertices</td></tr>
        </tbody>
      </table>
      <p>
        Run any kernel to the end in the explorer and it states its total against the bound. On the chain, the
        difference between d·n and n + m is the difference between n² and n.
      </p>
      <Aside>
        <p>
          Slip on p. 439: "the threads whose vertices are not in the <i>current</i> level do not do anything" means the
          previous level, as line 05 of Fig. 18.6 shows. The pull bound there also has an unbalanced parenthesis.
        </p>
      </Aside>
      <Ask q={questionsS183['ch18/18.3/q-chain']} />

      <h2>Frontiers</h2>
      <p>
        The fix is to launch threads only for the vertices labelled in the last call. Those threads build, as they go,
        the list of vertices they label, the frontier, and the next launch gets one thread per entry. Fig. 18.12 takes{' '}
        <code>prevFrontier</code> with <code>numPrevFrontier</code> entries, and <code>currFrontier</code> with a
        counter <code>numCurrFrontier</code> in global memory. To append, a thread does <code>fetch_add(1)</code> on the
        counter, which returns the old value, its own slot, and stores the vertex there (lines 12 to 16). That is
        chapter 12's unstable filter: the entries arrive in whatever order the atomics happen.
      </p>
      <Source section="18.5" pages="439–440" figs="Figs. 18.12, 18.13" />
      <p>
        Milestone M5 uses the same code shape to compact the tokens that survive a top-p cut into a dense list: claim a
        slot with one atomic, then store the token id.
      </p>
      <Beyond title="Bridge (ours)">
        <p>
          The sampling kernels are not in the book; <a href={href('ch12', 'labs')}>chapter 12's labs</a> build the M5
          candidate list.
        </p>
      </Beyond>
      <p>
        The flag is gone: the host stops when a launch leaves the frontier empty. A search of d levels still takes d + 1
        launches, the last one finding nothing, and each needs the host to read the frontier size back first. The
        animation puts the two kernels and the launches side by side on a wiring grid.
      </p>
      <Simplification>
        <p>
          The animation counts one thread per cell, without rounding up to whole blocks, and gives no times. Its grid is
          our own layout in the spirit of Fig. 18.5, not the book's figure.
        </p>
      </Simplification>
      <WavefrontScene />

      <h2>Exactly once</h2>
      <p>
        Without frontiers a duplicate label was harmless. With them it is not: two threads that both see a neighbour as{' '}
        <code>UNVISITED</code> both append it, and the next launch expands it twice. So checking and labelling must be
        one indivisible step. Compare-and-swap is that step: compare <code>level[v]</code> with <code>UNVISITED</code>
        and, only if they are equal, write <code>currLevel</code>, reporting whether it did. Fig. 18.14 wraps it in{' '}
        <code>visitVertexAtomically</code>, and of all the threads that race for a vertex exactly one gets true.
      </p>
      <Source section="18.5" pages="440–441" figs="Fig. 18.14" />
      <p>
        Three details. The expected value is passed by reference, so it must be a variable, not the constant{' '}
        <code>UNVISITED</code>. A failed compare-and-swap writes the value it found into that variable, so a variable
        reused for the next neighbour no longer holds <code>UNVISITED</code>; Fig. 18.14 declares it fresh on every
        call, which is right. And relaxed ordering on success and on failure is valid: nothing else needs ordering
        against this one location.
      </p>
      <Source section="18.5" pages="441–442" />
      <Today checked="2026-11-20 (notes/today/ch18-facts.md A6)">
        <p>
          libcu++ declares the expected parameter as a non-const reference. After a failure it holds the value that was
          found, as in standard C++. Relaxed for both orders is allowed.
        </p>
      </Today>
      <p>
        The book expects moderate contention on the compare-and-swap, since only threads with a common neighbour meet
        there, and high contention on the counter, which every appending thread hits. That counter is the next page's
        problem.
      </p>
      <Source section="18.5" pages="442" />
      <Aside>
        <p>
          Slip on p. 441: "the vertex-centric push kernel without frontiers in Fig. 18.12" means Fig. 18.6, whose lines
          09 and 10 are the check and the label.
        </p>
      </Aside>
      <Ask q={questionsS183['ch18/18.3/q-why-cas']} />
      <p>
        The stepper takes two or three threads of one level to a shared neighbour and lets you choose who moves when.
        Start with the plain check and an append: find an order that puts a vertex in the frontier twice. Then switch to
        compare-and-swap and press <i>Try every order</i>. Then hoist <code>e</code> above the loop and run the forward
        order on the trap scenario.
      </p>
      <Simplification>
        <p>
          One step is one statement, and it is indivisible: a load, a store, a compare-and-swap, or an append (a{' '}
          <code>fetch_add</code> and the store into its slot). Every load sees the latest store; there are no
          memory-order effects. A weak compare-and-swap fails spuriously only where you make it.
        </p>
      </Simplification>
      <VisitStepper />
      <Ask q={questionsS183['ch18/18.3/q-stale']} label="Bug hunt" />

      <h2>On HIP and Metal</h2>
      <p>
        The same compare-and-swap is spelled three ways, and it is not equally strong in all three. A strong
        compare-and-swap fails only when the values differ. A weak one may also fail when they are equal. Fig. 18.14
        relies on the strong form: a spurious failure would leave the vertex for a later level, or for none.
      </p>
      <Beyond>
        <p>
          The book uses CUDA only. Everything here about HIP and Metal is from the facts file and was not run on AMD or
          Apple hardware.
        </p>
      </Beyond>
      <p>
        On CUDA, <code>cuda::atomic_ref::compare_exchange_strong</code> is strong. On HIP the same line compiles, but
        libhipcxx implements it at block and device scope with the weak built-in; the legacy <code>atomicCAS</code> is
        strong and returns the old value. The course's <code>.cu</code> files therefore use <code>atomicCAS</code>{' '}
        under <code>__HIPCC__</code>, as lab 12.1 uses legacy atomics. Metal has only{' '}
        <code>atomic_compare_exchange_weak_explicit</code>. The portable form is the retry loop{' '}
        <code>while (!cas(e) &amp;&amp; e == UNVISITED) {'{}'}</code>, which costs nothing when no failure is spurious.
      </p>
      <Today checked="2026-11-20 (notes/today/ch18-facts.md A9, B4, C4, C5)">
        <p>
          Whether AMD or Apple GPUs ever fail a weak compare-and-swap spuriously is not documented: unverified. The
          sentinel's full spelling is <code>cuda::std::numeric_limits&lt;unsigned int&gt;::max()</code> from{' '}
          <code>&lt;cuda/std/limits&gt;</code>; MSL has <code>numeric_limits&lt;uint&gt;::max()</code> and{' '}
          <code>UINT_MAX</code>.
        </p>
      </Today>
      <VisitOnBackends />
      <table className="data wide" data-backends style={{ maxWidth: '58rem', tableLayout: 'fixed', width: '100%', overflowWrap: 'anywhere' }}>
        <colgroup>
          <col style={{ width: '19%' }} />
          <col style={{ width: '27%' }} />
          <col style={{ width: '27%' }} />
          <col style={{ width: '27%' }} />
        </colgroup>
        <thead>
          <tr><th>Feature</th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>Visit-once compare-and-swap</td>
            <td><TC>compare_exchange_strong</TC> on an <TC>atomic_ref</TC> (strong); legacy <TC>atomicCAS</TC></td>
            <td>libhipcxx's strong form lowered to the weak built-in at block and device scope; legacy <TC>atomicCAS</TC> strong</td>
            <td><TC>atomic_compare_exchange_weak_explicit</TC> only; spurious failure not documented either way</td>
          </tr>
          <tr>
            <td>Expected value after a failed compare-and-swap</td>
            <td>overwritten with the current value</td>
            <td>same</td>
            <td>same</td>
          </tr>
          <tr>
            <td>Global frontier counter</td>
            <td><TC>fetch_add(1, relaxed)</TC> on an <TC>atomic_ref&lt;…, device&gt;</TC></td>
            <td>same, through libhipcxx</td>
            <td><TC>atomic_fetch_add_explicit</TC> on a <TC>device atomic_uint*</TC></td>
          </tr>
          <tr>
            <td>Max-unsigned sentinel</td>
            <td><TC>numeric_limits&lt;unsigned int&gt;::max()</TC> from <TC>cuda::std</TC></td>
            <td>same, through libhipcxx; or <TC>UINT_MAX</TC></td>
            <td><TC>numeric_limits&lt;uint&gt;::max()</TC> or <TC>UINT_MAX</TC></td>
          </tr>
        </tbody>
      </table>
      <Ask q={questionsS183['ch18/18.3/q-weak-loop']} />
      <p>
        One counter that every thread hits, and one launch per level: those are the two costs left. The{' '}
        <a href={href('ch18', '18.6')}>next page</a> removes the first with a queue private to each block and the second
        with a barrier across the grid.
      </p>
    </>
  );
}
