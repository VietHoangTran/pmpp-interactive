import type { Question } from '../../components/Ask';
import { Beyond, Source, Today, type Tier } from '../../components/notes';
import {
  BackendsAndEmulator,
  LabFiles,
  LabPrediction,
  LabResults,
  LabsIntro,
  LabsPage,
  ReferenceSolution,
  RunningLabs,
  VerificationStatus,
  defineLabs,
  measure,
  msCol,
  numCol,
  rawCol,
  useLabs,
  type BenchSet,
  type Col,
} from '../../components/LabKit';
import { fmt, labStatus, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 18's own content.
// No question on this page: the lab is the exercise. The three predictions are stored, not
// graded, under ch18/labs/p-lab1-floor, ch18/labs/p-lab1-rmat and ch18/labs/p-lab1-coop.
// The benchmark rows are a contract with labs/ch18/lab1_frontier/main.cpp: names push,
// frontier, privatised, cooperative and floor; graph 'grid' or 'rmat'; seconds, levels,
// us_per_level and medges_per_s.
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch18', [{ num: '18.1', dir: 'lab1_frontier', name: 'frontier BFS' }]);
const [l1] = LABS;

/** Identifier colours in the reference solution: the graph and frontiers in global memory, the private queue in shared. */
const TIERS: Record<string, Tier> = {
  srcPtrs: 'global', dst: 'global', level: 'global', prevFrontier: 'global', currFrontier: 'global', numCurrFrontier: 'global',
  currFrontier_s: 'shared', numCurrFrontier_s: 'shared', currFrontierStartIdx: 'shared',
  i: 'reg', vertex: 'reg', vertexId: 'reg', edge: 'reg', neighbor: 'reg', currFrontierIdx_s: 'reg', currFrontierIdx: 'reg',
  unvisited: 'reg', expected: 'reg', visitSuccess: 'reg', count: 'reg', j: 'reg', gridRank: 'reg', gridThreads: 'reg',
};

/** The benchmark's two graphs, in the driver's order. */
export const GRAPHS = ['grid', 'rmat'] as const;
/** The benchmark's searches, in the driver's order. */
export const SEARCHES = ['push', 'frontier', 'privatised', 'cooperative', 'floor'] as const;

// ---------------------------------------------------------------- reading the benchmark rows
const row = (s: BenchSet, name: string, graph: string): BenchRow | undefined =>
  s.rows.find((r) => r.name === name && r.graph === graph);
const secs = (r: BenchRow | undefined) => num(r?.seconds);
const ms = (r: BenchRow | undefined) => {
  const v = secs(r);
  return v ? `${fmt(v * 1000)} ms` : '?';
};
const levelsOf = (r: BenchRow | undefined) => (num(r?.levels) != null ? `${num(r?.levels)!.toLocaleString('en-US')} levels` : '');

/** Prediction 1: on 'grid', floor time over privatised time. */
const floorShare = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const fl = row(s, 'floor', 'grid');
    const pr = row(s, 'privatised', 'grid');
    const a = secs(fl);
    const b = secs(pr);
    if (!a || !b) return null;
    return { value: a / b, detail: `${ms(fl)} for the empty levels against ${ms(pr)} for the privatised search, ${levelsOf(pr)}` };
  });

/** Prediction 2: on 'rmat', frontier time over privatised time. */
const rmatRelief = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const fr = row(s, 'frontier', 'rmat');
    const pr = row(s, 'privatised', 'rmat');
    const a = secs(fr);
    const b = secs(pr);
    if (!a || !b) return null;
    return { value: a / b, detail: `${ms(fr)} frontier against ${ms(pr)} privatised, ${levelsOf(pr)}` };
  });

/** Prediction 3, grid half: privatised time over cooperative time. Only CUDA and HIP have a cooperative row. */
const coopGain = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const pr = row(s, 'privatised', 'grid');
    const co = row(s, 'cooperative', 'grid');
    const a = secs(pr);
    const b = secs(co);
    if (!a || !b) return null;
    const pr2 = secs(row(s, 'privatised', 'rmat'));
    const co2 = secs(row(s, 'cooperative', 'rmat'));
    const rmat = pr2 && co2 ? `; on rmat the same ratio is ${fmt(pr2 / co2)}` : '; no rmat pair in this run';
    return { value: a / b, detail: `${ms(pr)} privatised against ${ms(co)} cooperative on grid${rmat}` };
  });

const MISSING = (
  <>
    This needs the benchmark rows of lab 18.1 from a real device. Run <code>python3 labs/pmpp.py run 18.1 --bench</code>{' '}
    once your kernels pass (or with <code>--solution</code> to time the reference). The emulators are never benchmarked.
  </>
);
const MISSING_COOP = (
  <>
    This needs a <code>cooperative</code> row, which only a CUDA or HIP device that supports cooperative launch can give,
    once part B passes: <code>python3 labs/pmpp.py run 18.1 --bench</code> there. Metal has no part B, and a run on
    Metal alone leaves this empty.
  </>
);

// ---------------------------------------------------------------- results table
const COLS: Col[] = [
  { head: 'Search', cell: (r) => String(r.name ?? '') },
  { head: 'Graph', cell: (r) => String(r.graph ?? '') },
  numCol('Vertices', 'vertices'),
  numCol('Edges', 'edges'),
  rawCol('Levels', 'levels'),
  msCol,
  numCol('µs per level', 'us_per_level'),
  numCol('Million edges/s', 'medges_per_s'),
];

/** One line on lab 18.1 on the race backend, which the status table does not show. */
function RaceStatus() {
  const { reports, loading } = useLabs();
  const run = reports.race;
  const line = (variant: 'solution' | 'work') => {
    const rs = (run?.results.filter((x) => x.lab === l1.id && x.variant === variant) ?? []) as (RunResult & { tsan?: unknown[] })[];
    const tests = rs.flatMap((x) => x.tests);
    const found = rs.reduce((n, x) => n + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0);
    return { s: labStatus(run, l1.id, variant), tests, passed: tests.filter((t) => t.pass).length, found };
  };
  const sol = line('solution');
  const mine = line('work');
  return (
    <p data-race-status={loading ? 'loading' : sol.s} data-race-work={loading ? 'loading' : mine.s}>
      {loading ? (
        'Reading reports.'
      ) : !run ? (
        'No run on the race backend on this machine yet.'
      ) : (
        <>
          {sol.s === 'none'
            ? 'On this machine the reference solution has no run on the race backend yet.'
            : sol.s === 'pass'
              ? `On this machine the reference solution passed all ${sol.tests.length} tests on the race backend with ${sol.found ? `${sol.found} ThreadSanitizer report(s)` : 'no race reported'}: emulator-checked on this flavour.`
              : `On this machine the reference solution failed on the race backend (${sol.passed} of ${sol.tests.length} tests): unverified there.`}{' '}
          {mine.s === 'none'
            ? 'Your own kernels have no recorded run there, or are still the untouched starter.'
            : mine.s === 'pass'
              ? `Your own kernels passed all ${mine.tests.length} tests there with ${mine.found ? `${mine.found} ThreadSanitizer report(s): read them` : 'no race reported'}.`
              : `Your own kernels passed ${mine.passed} of ${mine.tests.length} tests there, with ${mine.found} ThreadSanitizer report(s) in the run.`}
        </>
      )}
    </p>
  );
}

/** What each backend calls the pieces of the lab (notes/today/ch18-facts.md, "Where the backends really differ"). */
function BackendsTable() {
  return (
    <table className="data wide" style={{ maxWidth: '58rem' }} data-table="backends">
      <thead>
        <tr><th>In this lab</th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>Visit exactly once</td>
          <td><code>atomic_ref::compare_exchange_strong</code>, a fresh expected value per call</td>
          <td>legacy <code>atomicCAS(&amp;level[v], UNVISITED, currLevel) == UNVISITED</code>, which is strong</td>
          <td><code>atomic_compare_exchange_weak_explicit</code> in a loop that retries while <code>expected</code> still holds <code>UNVISITED</code></td>
        </tr>
        <tr>
          <td>Private counter</td>
          <td><code>atomic_ref&lt;unsigned, thread_scope_block&gt;</code> on a <code>__shared__</code> variable</td>
          <td>legacy <code>atomicAdd</code> on the same variable</td>
          <td><code>threadgroup atomic_uint</code>, zeroed with <code>atomic_store_explicit</code></td>
        </tr>
        <tr>
          <td>Reset on overflow (line 29)</td>
          <td><code>store</code> through the same <code>atomic_ref</code>, or a clamp after the barrier</td>
          <td><code>atomicExch</code></td>
          <td><code>atomic_store_explicit</code>: the plain store cannot be written</td>
        </tr>
        <tr>
          <td>Sentinel</td>
          <td colSpan={2}><code>UINT_MAX</code> from <code>&lt;climits&gt;</code> (the emulators have no <code>&lt;cuda/std/limits&gt;</code>)</td>
          <td><code>numeric_limits&lt;uint&gt;::max()</code></td>
        </tr>
        <tr>
          <td>Part B: grid barrier</td>
          <td><code>this_grid().sync()</code>, <code>&lt;cooperative_groups.h&gt;</code>, 64-bit ranks</td>
          <td>same names, <code>&lt;hip/hip_cooperative_groups.h&gt;</code>, 32-bit ranks</td>
          <td>none: one dispatch per level</td>
        </tr>
        <tr>
          <td>Part B: launch</td>
          <td><code>cudaLaunchCooperativeKernel</code>, six arguments</td>
          <td>the same call through <code>pmpp/vendors/hip.h</code>; six are required</td>
          <td>none</td>
        </tr>
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS} tiers={TIERS}>
      <LabsIntro>
        One lab, in <code>labs/ch18/lab1_frontier/</code>. It is optional, as this chapter is condensed, and it adds
        nothing to the through-line project. It turns the corrected kernels of page{' '}
        <a href={href('ch18', '18.6')}>18.6</a> into code you run on every backend, and its benchmark asks the question
        that page leaves open on your own GPU: how much of a search with many small levels is the fixed cost of a level.
      </LabsIntro>
      <RunningLabs example="18.1" />
      <BackendsAndEmulator />
      <VerificationStatus />

      {/* ------------------------------------------------------------ 18.1 */}
      <h2>Lab 18.1: frontier BFS with a privatised queue</h2>
      <p>
        Part A, on every backend, is the search of page 18.6 with one launch per level. You write the visit, a
        compare-and-swap that turns <code>UNVISITED</code> into the current level for exactly one thread, and the
        level kernel: one thread per vertex of the previous frontier, a private frontier and counter per block in shared
        memory, an overflow path to the public frontier when the private one is full, one device-scope atomic per block
        to reserve its range, and a commit in which neighbouring threads write neighbouring addresses.
      </p>
      <Source section="18.5, 18.6" pages="440–444" figs="Figs. 18.14, 18.15" />
      <p>
        Part B, optional, runs on CUDA, HIP and the emulators. The level loop moves into the kernel: a grid sized so that
        every block is resident, each thread striding through the previous frontier, and three grid barriers at the end
        of each level. You also write the sizing lines of the launcher, an occupancy query times the number of SMs, and
        the cooperative launch.
      </p>
      <Source section="18.7" pages="445–448" figs="Fig. 18.17" />
      <p>
        Metal has no part B: it documents no grid barrier, no cooperative launch and no occupancy query from which to size
        a resident grid. On a Mac the search stays at one dispatch per level.
      </p>
      <Today checked="2026-11-20 (notes/today/ch18-facts.md C1; ch04-facts C3, C5)">
        <p>No grid-wide barrier, no cooperative launch and no core count or occupancy query are documented for Metal.</p>
      </Today>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <>
              <code>visitVertexAtomically</code> and <code>bfsPrivatisedKernel</code> (part A); <code>bfsCooperativeKernel</code>{' '}
              and the marked lines of <code>launchBfsCooperative</code> (part B). The per-level launcher{' '}
              <code>launchBfsPrivatised</code> is written for you
            </>,
          ],
          [
            'kernel.metal',
            <>
              <code>visit_vertex</code> and <code>bfs_privatised</code>, with <code>level</code> as{' '}
              <code>device atomic_uint*</code> and the private counter a <code>threadgroup atomic_uint</code>
            </>,
          ],
          [
            'launch.mm',
            <>
              provided: one command buffer per level, <code>waitUntilCompleted</code>, then a read of the count. You need not
              edit it
            </>,
          ],
        ]}
      />
      <p>
        Read-only: <code>provided/ref_kernels.cu</code>, <code>provided/ref_kernels.metal</code> and{' '}
        <code>provided/ref_launch.mm</code>, the benchmark's baselines: <code>push</code> (Fig. 18.6, every vertex at
        every level, with relaxed atomics in place of the page's benign race), <code>frontier</code> (Figs. 18.12 and
        18.14, one public atomic per vertex found) and <code>floor</code>, an empty kernel. And <code>main.cpp</code>: the
        graphs, the sequential BFS, the tests and the benchmark. <code>BLOCK_DIM</code> is 128 and{' '}
        <code>PRIVATE_FRONTIER_CAPACITY</code> is 256, larger than the block, so the commit loop has to go round more than
        once when a block finds many vertices.
      </p>

      <h3>Where the lab differs from the page</h3>
      <p>
        <b>Line 29 is an atomic store.</b> On overflow, Fig. 18.15 puts the private counter back to the capacity with a
        plain store while other threads of the block may be doing <code>fetch_add</code> on it (line 35 of Fig. 18.17).
        That is a data race, so undefined behaviour, whatever the hardware does. Store through the same{' '}
        <code>atomic_ref</code>, relaxed, or drop the store and clamp the count with <code>min</code> after the barrier.
      </p>
      <Today checked="2026-11-20 (notes/today/ch18-facts.md A7, E1)">
        <p>
          A plain access racing an atomic one is a data race by NVIDIA's own definition (ch09 A2). That the atomic store
          keeps the counts right is our enumeration of every interleaving for small cases, not a vendor statement.
        </p>
      </Today>
      <p>
        <b>UINT_MAX for the sentinel.</b> The page's <code>cuda::std::numeric_limits&lt;unsigned&gt;</code> names the
        trait without <code>::max()</code>. The <code>.cu</code> files use <code>UINT_MAX</code> from{' '}
        <code>&lt;climits&gt;</code>, the same value, because the course's emulators ship no{' '}
        <code>&lt;cuda/std/limits&gt;</code>.
      </p>
      <p>
        <b>Shared variables at the top.</b> Fig. 18.17 declares its <code>__shared__</code> variables inside the level
        loop. They live as long as the block either way; declare them once, at the top, and let the reset at the start of
        each level empty the private frontier.
      </p>
      <p>
        <b>Six arguments, 32-bit ranks.</b> Pass all six arguments to <code>cudaLaunchCooperativeKernel</code>, ending{' '}
        <code>kernelArgs, 0, 0</code>: CUDA defaults the last two, HIP does not. And cast{' '}
        <code>grid.thread_rank()</code> and <code>grid.num_threads()</code> to <code>unsigned int</code> once, since they
        are 64-bit on CUDA and 32-bit on HIP.
      </p>
      <Today checked="2026-11-20 (notes/today/ch18-facts.md A1, A4, B1, B2)">
        <p>
          The four-argument call compiles for CUDA only. AMD's own example passes <code>0</code> as the argument array of a
          kernel with parameters: do not copy it.
        </p>
      </Today>
      <p>
        <b>Guard words on the public frontier.</b> No listing checks the public frontier's capacity, and the page shows
        no allocation. Here each public frontier holds <code>numVertices</code> entries, enough when each vertex is added
        once, and the driver puts guard words after both buffers and checks them after every search.
      </p>
      <p>
        <b>Raw pointers.</b> <code>srcPtrs</code>, <code>dst</code> and <code>numVertices</code> are separate arguments;
        the book's <code>CSRGraph</code> struct is never defined. Every listing in the chapter is called{' '}
        <code>bfs_kernel</code>; the names here are ours.
      </p>

      <h3>The same kernels on other backends</h3>
      <p>
        The search itself is the same on all three. What differs is the compare-and-swap's strength, how a counter is
        declared, and whether part B exists at all.
      </p>
      <Beyond>
        <p>
          The book uses CUDA only. Everything in this table about HIP and Metal is from notes/today/ch18-facts.md, checked
          2026-11-20. The HIP branches of the lab have not been compiled for HIP or run on AMD hardware.
        </p>
      </Beyond>
      <BackendsTable />
      <p>
        On HIP, <code>cuda::atomic_ref</code> compiles through libhipcxx, but its <code>compare_exchange_strong</code> is
        implemented with the weak built-in at block and device scope. That is why the <code>.cu</code> files write the
        legacy calls under <code>#if defined(__HIPCC__)</code>, as lab 12.1 does. Whether AMD or Apple GPUs ever fail a
        weak compare-and-swap spuriously is not documented: unverified either way, which is why the Metal visit loops.
      </p>
      <Today checked="2026-11-20 (notes/today/ch18-facts.md B4, C4; infra-facts B1, B2)">
        <p>
          libhipcxx lowers the strong form to <code>__hip_atomic_compare_exchange_weak</code>; HIP's{' '}
          <code>atomicCAS</code> is strong. MSL has no strong form.
        </p>
      </Today>

      <h3>What the tests check</h3>
      <p>
        Run <code>python3 labs/pmpp.py run 18.1 --backend emu</code> first. Every case compares <code>level</code> with a
        sequential BFS in <code>main.cpp</code> exactly; unreached vertices must stay <code>UNVISITED</code>. For the
        per-level search, each level's frontier size must equal the sequential count of vertices at that level, which
        catches a vertex added twice and a vertex lost, and the number of launches must match. The guard words after both
        public frontiers must be untouched.
      </p>
      <p>
        The graphs: the book's graph from roots 0 and 2, which must give Fig. 18.4's levels; the Exercise 1 graph from 0,
        in our reading of the drawing on p. 450; a single vertex; a root with no out-edges; two components; self-loops
        and repeated edges; a chain; a grid with 4 neighbours per vertex. Then the cases built to break one step each.
        "Fan-out" has 128 children find 512 grandchildren in one block against 256 private slots, so many threads
        overflow at once and the commit has more entries than the block has threads. "Fan-out shared" makes children
        compete for the same grandchildren. A hub of degree 5,000 overflows alone. The "stale-expected trap", 0 to 1 and
        2, 1 to 3, 2 to 3 and 4, makes vertex 2's thread lose vertex 3 and still have to win vertex 4. Two R-MAT graphs
        from fixed seeds close the list. Part B runs the same graphs through <code>launchBfsCooperative</code>; on the
        emulator that is 8 blocks of 128 threads, its limit of 1,024 threads for a cooperative grid.
      </p>
      <p>
        Each backend sees different mistakes. The emulator runs one thread at a time in three orders under
        AddressSanitizer: it catches an expected value reused across calls, a missing capacity check, a commit written as
        an <code>if</code>, and most missing barriers. It cannot see a missing compare-and-swap or the page's plain store
        on line 29, because nobody is interrupted between a check and a store, and some missing barriers happen to give
        the right answer under its schedule. The <code>race</code> flavour runs the threads concurrently under
        ThreadSanitizer and names both source lines of each unsynchronised pair: run{' '}
        <code>python3 labs/pmpp.py run 18.1 --backend race</code> before you trust a pass. A GPU shows duplicates or lost
        vertices only when its schedule produces them, and a missing barrier can pass there.
      </p>
      <RaceStatus />
      <p>
        On Metal, part A runs as one compute dispatch per level, each in its own command buffer, waited on before the
        count is read back. The dispatches run in order and each sees the previous one's writes, because the course's
        runtime uses an <code>MTLCommandQueue</code> with tracked buffers and a plain compute encoder. Part B's tests are
        skipped there with a message, as they are on a CUDA or HIP device that reports no cooperative launch, and while{' '}
        <code>launchBfsCooperative</code> still returns <code>BFS_COOP_NOT_WRITTEN</code>.
      </p>
      <Today checked="2026-11-20 (notes/today/ch18-facts.md C2)">
        <p>
          Serial dispatch order and hazard tracking hold on an <code>MTLCommandQueue</code>; an{' '}
          <code>MTL4CommandQueue</code> tracks no hazards.
        </p>
      </Today>

      <h3>The experiment</h3>
      <p>
        Do this once your kernels pass, and before you run <code>--bench</code>. The benchmark builds two graphs with
        about the same number of edges. <code>grid</code> is 1024 × 1024 with 4 neighbours per vertex, searched from a
        corner: 2,046 levels, none larger than 1,024 vertices, a road network in miniature. <code>rmat</code> is an R-MAT
        graph from a fixed seed: a handful of levels with large frontiers, a social network in miniature. For each it
        times whole searches, best of 5 after an untimed run, with <code>level</code> reset outside the timing, and checks
        every result before it reports a row: <code>push</code>, <code>frontier</code>, your <code>privatised</code>, your{' '}
        <code>cooperative</code> (CUDA and HIP only) and <code>floor</code>.
      </p>
      <Beyond>
        <p>
          The R-MAT generator and its parameters are ours and are in main.cpp; so is the floor. The book gives no
          measurement for any of this.
        </p>
      </Beyond>
      <p>
        <code>floor</code> is the host loop of the per-level search wrapped around an empty kernel, once per level of
        that graph: zero the counter, launch, read 4 bytes back; on Metal, an empty dispatch, a wait and a read. It is the
        fixed cost of a level with no search in it. The book says that with small frontiers this cost may dominate, and
        with large ones it is negligible.
      </p>
      <Source section="18.7" pages="445" />
      <p>
        Privatisation is meant to take contention off the one public counter, which every appending thread of the
        unprivatised kernel hits. The compiler may already combine those atomics within a warp, which would shrink what
        privatisation gains. Write down three numbers.
      </p>
      <Today checked="2026-11-20 (notes/today/ch18-facts.md A8)">
        <p>
          NVIDIA states automatic warp aggregation only in a 2017 blog post; AMD's compiler does it for uniform addresses;
          Apple documents nothing.
        </p>
      </Today>
      <LabPrediction
        id="ch18/labs/p-lab1-floor"
        prompt={
          <>
            On <code>grid</code>, the share of your privatised search's time that the empty levels account for:{' '}
            <code>floor</code> time divided by <code>privatised</code> time.
          </>
        }
        unit="(floor time / privatised time)"
        measure={floorShare}
        missing={MISSING}
        reading={
          <>
            Close to 1 means the search is mostly the cost of having a level at all, which is the p. 445 claim about
            road networks. Both rows run the same number of levels, so the difference between them is the search's own
            work.
          </>
        }
      />
      <LabPrediction
        id="ch18/labs/p-lab1-rmat"
        prompt={
          <>
            On <code>rmat</code>, how many times faster <code>privatised</code> is than <code>frontier</code>:{' '}
            <code>frontier</code> time divided by <code>privatised</code> time.
          </>
        }
        unit="(frontier time / privatised time)"
        measure={rmatRelief}
        missing={MISSING}
        reading={
          <>
            Above 1, the private queues paid for their barriers and shared-memory traffic by taking contention off the one
            public counter. Below 1, they did not, on this graph and this device.
          </>
        }
      />
      <LabPrediction
        id="ch18/labs/p-lab1-coop"
        prompt={
          <>
            On a CUDA or HIP device, on <code>grid</code>, how many times faster <code>cooperative</code> is than{' '}
            <code>privatised</code>. Then say, without recording it, what you expect on <code>rmat</code>.
          </>
        }
        unit="(privatised time / cooperative time)"
        measure={coopGain}
        missing={MISSING_COOP}
        reading={
          <>
            The cooperative grid is capped at what is resident, so on <code>rmat</code>'s large frontiers each thread
            strides through many vertices, and three grid barriers per level replace one launch. Which of those costs
            wins is what the two graphs separate.
          </>
        }
      />
      <LabResults lab={l1} cols={COLS} />
      <p>
        How to read the table. "µs per level" divides each time by the number of launches, the levels after the root
        plus the last launch, which finds nothing. "Million edges/s" counts the out-edges of every reached vertex, which
        every search here examines. Times for the same search vary from one run to the next, so run the benchmark twice
        before you trust a small difference between two searches.
      </p>

      <h3>Where this sits under what you know</h3>
      <p>
        Token generation is the grid graph's shape: many short steps, each launched by the host, each doing little work.
        The floor row is three costs per level: a launch, a memset of the counter, and a blocking read of the count back
        to the host. A CUDA graph pays the launch part once for the whole graph instead of per launch; it does not
        remove the read-back, which is what stops a host loop like this from being captured at all. Taking that out
        needs the level loop on the device: a conditional WHILE node (CUDA 12.3 and later) or the cooperative kernel
        of lab 18.1. And the compaction in your level kernel, claim a slot with an atomic and store, is the shape of
        collecting the survivors of a top-p cut.
      </p>
      <Beyond>
        <p>
          The decoding and sampling readings are ours, not the book's; page 18.6 gives llama.cpp's and PyTorch's own
          answers to launch overhead. What graphs and WHILE nodes do is NVIDIA's account, and the read-back's effect
          on capture PyTorch's (notes/today/ch18-facts.md A10, D5, checked 2026-11-20).
        </p>
      </Beyond>

      <h3>Go further</h3>
      <p>
        Exercise 3 asks for Fig. 18.17 with one grid barrier per level: a counter per level, all zeroed before the launch.
        Page 18.6 shows one answer, checked on the emulator. Write it into your part B and time it against your
        three-barrier version on <code>grid</code>. And swap your line-29 store for the clamp after the barrier, run{' '}
        <code>race</code> on both, then put the book's plain store back and read what ThreadSanitizer says.
      </p>
      <Source section="18.10" pages="450" />
      <ReferenceSolution lab={l1} />
    </LabsPage>
  );
}
