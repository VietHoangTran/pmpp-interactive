import { Predict, type Question } from '../../components/Ask';
import { Beyond, Source, Today, type Tier } from '../../components/notes';
import {
  BackendsAndEmulator,
  LabFiles,
  LabPrediction,
  LabResults,
  LabsIntro,
  LabsPage,
  RaceStatus,
  ReferenceSolution,
  RunningLabs,
  VerificationStatus,
  benchRow,
  defineLabs,
  measure,
  msCol,
  numCol,
  rawCol,
  useLabs,
  type BenchSet,
} from '../../components/LabKit';
import { fmt, labStatus, num, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 4's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch04', [
  { num: '4.1', dir: 'lab1_launchconfig', name: 'launch from queries (M1)' },
  { num: '4.2', dir: 'lab2_divergence', name: 'divergence' },
  { num: '4.3', dir: 'lab3_barrierhunt', name: 'barrier hunt' },
  { num: '4.4', dir: 'lab4_blocksize', name: 'block size' },
]);
const [l1, l2, l3, l4] = LABS;

// Lab 3.3 is the other half of milestone M1.
const M1_FIRST_HALF = 'ch03/lab3_naive_gemm';

// Identifier colours for the reference solutions: buffers are global memory, values a
// thread computes for itself are registers. Only names that mean one thing in all three files.
const tiers = (global: string[], reg: string[]): Record<string, Tier> => ({
  ...Object.fromEntries(global.map((n) => [n, 'global' as Tier])),
  ...Object.fromEntries(reg.map((n) => [n, 'reg' as Tier])),
});
const LINEAR_TIERS = tiers(['X', 'W', 'Y'], []);
const DIVERGENCE_TIERS = tiers(['which', 'trips'], []);
const BARRIER_TIERS = tiers([], ['mine', 'valid', 'base']);
const GATHER_TIERS = tiers(['table', 'idx'], ['sum']);

// ---------------------------------------------------------------- what each prediction is measured against
const ms = (s: number) => `${fmt(s * 1000)} ms`;
const shape = (r: BenchRow | undefined, x: string, y: string) => (r ? `${String(r[x])} × ${String(r[y])}` : '');

/** Lab 4.1: at batch 1, the derived configuration's speed over fixed 16 × 16 blocks. */
const derivedOverFixed = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const v = num(benchRow(s, 'speedup', { batch: 1 })?.derived_over_fixed);
    const derived = s.rows.find((r) => r.name === 'linear' && r.config === 'derived' && r.batch === 1);
    const fixed = s.rows.find((r) => r.name === 'linear' && r.config === 'fixed16' && r.batch === 1);
    const a = num(derived?.seconds);
    const b = num(fixed?.seconds);
    if (!v || !a || !b) return null;
    return { value: v, detail: `${ms(a)} with your ${shape(derived, 'block_x', 'block_y')} blocks, ${ms(b)} with 16 × 16, batch 1` };
  });

/** Lab 4.1: the width the device reported (warpSize, or the pipeline's threadExecutionWidth on Metal). */
const reportedWidth = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const queried = s.rows.filter((r) => r.name === 'queried');
    const row = queried.find((r) => r.device === 0) ?? queried[0];
    const metal = s.backend === 'metal';
    const v = num(metal ? row?.threadExecutionWidth : row?.warpSize);
    if (!v) return null;
    return {
      value: v,
      detail: metal ? 'threadExecutionWidth of the linear-layer pipeline' : 'warpSize of device 0, from your queryDevices',
    };
  });

const isMixed = (r: BenchRow) => String(r.pattern ?? '').startsWith('half of each');
const mixedRow = (s: BenchSet, name: string) => s.rows.find((r) => r.name === name && isMixed(r));

/** Lab 4.2: the branch with both paths in every warp, over the same mix sorted. */
const divergenceCost = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const r = mixedRow(s, 'branch');
    const v = num(r?.ratio_to_sorted);
    const mixed = num(r?.seconds);
    const sorted = num(r?.sorted_seconds);
    if (!r || !v || !mixed || !sorted) return null;
    return { value: v, detail: `${ms(mixed)} mixed, ${ms(sorted)} sorted, width ${String(r.width)}` };
  });

/** Lab 4.2: select over branch, on that same mixed pattern. */
const selectCost = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const sel = mixedRow(s, 'select');
    const br = mixedRow(s, 'branch');
    const v = num(sel?.ratio_to_branch);
    const a = num(sel?.seconds);
    const b = num(br?.seconds);
    if (!v || !a || !b) return null;
    return { value: v, detail: `${ms(a)} for the select, ${ms(b)} for the branch` };
  });

/** Lab 4.4: the slowest block size's time over the fastest's. */
const blockSizeSpread = (reports: Reports) =>
  measure(reports, l4, (s) => {
    const r = benchRow(s, 'summary');
    const v = num(r?.slowest_over_fastest);
    const noise = num(r?.largest_median_over_best);
    if (!r || !v) return null;
    return {
      value: v,
      detail:
        `slowest ${String(r.slowest_threads_per_block)} threads per block, fastest ${String(r.fastest_threads_per_block)}` +
        (noise ? `; the median run of one size was up to ${fmt(noise)} times its own best` : ''),
    };
  });

// ---------------------------------------------------------------- lab 4.3: the experiment of exercise 5
const BARRIER_ID = 'ch04/labs/p-lab3-barrier';
const OUTCOMES = [
  'Neither fails: both block sizes pass every launch.',
  'Only the block four warps wide fails.',
  'Only the block one warp wide fails.',
  'Both fail.',
] as const;
const GPU_BACKENDS = ['metal', 'cuda', 'hip'];

interface CaseResult {
  /** The test cases of one block size, as the driver named them. */
  cases: { name: string; pass: boolean; detail: string }[];
  failed: boolean;
}
interface BarrierRun {
  backend: string;
  device: string;
  date: string;
  untouched: boolean;
  one: CaseResult;
  four: CaseResult;
  outcome: number;
}
/** The two block sizes the experiment is about, from the last run of the reader's work/ on each GPU. */
function barrierRuns(reports: Reports): BarrierRun[] {
  return GPU_BACKENDS.flatMap((b) => {
    const run = reports[b];
    if (!run) return [];
    const tests = run.results.filter((x) => x.lab === l3.id && x.variant === 'work').flatMap((x) => x.tests);
    const pick = (label: string): CaseResult => {
      const cases = tests
        .filter((t) => t.case.includes(`(${label})`))
        .map((t) => ({ name: t.case, pass: t.pass, detail: t.detail ?? '' }));
      return { cases, failed: cases.some((c) => !c.pass) };
    };
    const one = pick('1 x width');
    const four = pick('4 x width');
    if (!one.cases.length || !four.cases.length) return [];
    return [{
      backend: b,
      device: String(run.device.name ?? b),
      date: run.generated.slice(0, 10),
      untouched: run.work_state?.[l3.id] === 'untouched',
      one,
      four,
      outcome: (one.failed ? 2 : 0) + (four.failed ? 1 : 0),
    }];
  });
}
/** "passed every launch" or the driver's own count of wrong launches. */
const verdict = (c: CaseResult) => {
  if (!c.failed) return 'passed every launch';
  const counts = c.cases
    .filter((x) => !x.pass)
    .map((x) => /wrong in \d+ of \d+ launches/.exec(x.detail)?.[0] ?? 'failed')
    .join('; ');
  return `failed (${counts})`;
};

function BarrierReveal({ committed }: { committed: number }) {
  const { reports, loading, reload } = useLabs();
  const runs = barrierRuns(reports);
  return (
    <>
      {runs.length ? (
        runs.map((r) => (
          <div key={r.backend} data-measured={r.outcome} data-backend={r.backend}>
            <p>
              On {r.device} ({r.backend}), the last run of your <code>work/</code> on {r.date}
              {r.untouched ? ', which was still the starter as shipped' : ''}: one warp wide,{' '}
              <span data-case="one">{verdict(r.one)}</span>; four warps wide, <span data-case="four">{verdict(r.four)}</span>.
            </p>
            <p>
              <b>{OUTCOMES[r.outcome]}</b>{' '}
              {r.untouched
                ? 'The starter still has all three defects, so this is not the experiment yet.'
                : r.outcome === committed
                  ? 'That is the outcome you predicted.'
                  : 'That is not the outcome you predicted.'}
            </p>
          </div>
        ))
      ) : loading ? (
        <p data-measured="loading">Reading reports.</p>
      ) : (
        <p data-measured="none">
          No run of your work for this lab on a GPU yet. Fix the three defects, delete the barrier, and run{' '}
          <code>python3 labs/pmpp.py run 4.3</code> on a machine with a GPU.{' '}
          <button className="quiet" onClick={reload}>
            Reload results
          </button>
        </p>
      )}
      <p className="muted">
        This reads the last run of your own <code>work/</code>. It is the experiment only if that run had the other two
        defects fixed and the barrier removed; with any defect still in, failures say nothing about the barrier. A
        block size that passed all its launches showed you the schedules your GPU happened to run, not that the kernel
        is correct.
      </p>
    </>
  );
}

/** A prediction with four outcomes, stored as the outcome's written index. Stored, not graded. */
function BarrierPrediction() {
  return (
    <Predict
      id={BARRIER_ID}
      options={[...OUTCOMES]}
      prompt={
        <>
          Before the experiment. With the other two defects fixed and the barrier removed, you run the tests on your GPU.
          What happens at the block one warp wide (the case marked <code>1 x width</code>) and at the block four warps
          wide (<code>4 x width</code>)?
        </>
      }
    >
      {(committed) => <BarrierReveal committed={committed} />}
    </Predict>
  );
}

// ---------------------------------------------------------------- milestone M1
function MilestoneStatus() {
  const { reports, loading } = useLabs();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const both = backends.filter(
    (b) => labStatus(reports[b], M1_FIRST_HALF, 'work') === 'pass' && labStatus(reports[b], l1.id, 'work') === 'pass',
  );
  const name = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  return (
    <p data-milestone={loading ? 'loading' : both.join(',') || 'none'}>
      {loading
        ? 'Reading reports.'
        : !backends.length
          ? 'No lab results are recorded on this machine yet.'
          : both.length
            ? `Your own labs 3.3 and 4.1 both pass on ${both.map(name).join(' and ')} on this machine: M1 is done there.`
            : 'Your own labs 3.3 and 4.1 do not both pass on any backend of this machine yet.'}
    </p>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS}>
      <LabsIntro>
        Four labs, each in its own directory under <code>labs/ch04/</code>. The first asks the device for its limits and
        sizes a launch from them, which finishes the project's first milestone. The second and the fourth measure what
        sections 4.5 and 4.7 can only count. The third is a bug hunt around a barrier.
      </LabsIntro>
      <RunningLabs
        example="4.2"
        alsoShownIn={
          <>
            sections <a href={href('ch04', '4.4')}>4.4</a>, <a href={href('ch04', '4.5')}>4.5</a>,{' '}
            <a href={href('ch04', '4.7')}>4.7</a> and <a href={href('ch04', '4.8')}>4.8</a>
          </>
        }
      />
      <BackendsAndEmulator />
      <p>
        The emulator has a second flavour, <code>race</code>, which lab 4.3 is also built for. It runs the emulated
        threads concurrently under ThreadSanitizer and reports two threads that touch the same element without
        synchronisation, with both source lines. Ask for it with <code>--backend race</code>;{' '}
        <code>python3 labs/pmpp.py doctor</code> says whether your compiler can build it. The status table below gives
        it a column of its own; the other tables on this page show the first flavour only.
      </p>
      <p>
        Two of this chapter's measurements, what a divergent warp costs and what block size does to a kernel that
        waits on memory, exist only on a GPU. The emulators check that labs 4.2 and 4.4 compute the right values, and
        nothing about their speed.
      </p>
      <VerificationStatus />

      {/* ------------------------------------------------------------ 4.1 */}
      <h2>Lab 4.1: ask the device, then size the launch</h2>
      <p>
        Lab 3.3 launched the naive GEMM with 16 × 16 blocks because the book's figures do. Here the device decides. You
        write the query loop of <a href={href('ch04', '4.8')}>section 4.8</a>, a pure function that turns what the
        device reports into a block and a grid, and the launch that uses both. The kernel is lab 3.3's, given complete
        and restated with a linear layer's names: <code>Y[b][o]</code> is the sum over <code>k</code> of{' '}
        <code>X[b][k] * W[k][o]</code>, row-major, batch first, one thread per output, the same argument order as{' '}
        <code>naiveGemmKernel</code>.
      </p>
      <Source section="4.8" pages="87–89" figs="the slot arithmetic is §4.7, pp. 85–86; the warp cut §4.4, p. 75" />
      <LabFiles
        lab={l1}
        files={[
          ['kernel.cu', <><code>queryDevices</code> (the loop of p. 88), <code>chooseLaunch</code> and <code>launchLinear</code></>],
          ['kernel.metal', <>the probe kernel <code>simd_probe</code>; the linear kernel is given</>],
          ['launch.mm', <><code>queryMetal</code>, <code>encodeLinear</code> and <code>encodeProbe</code></>],
        ]}
      />
      <p>
        A small header beside them, <code>launch_config.h</code>, holds the <code>DeviceInfo</code> and{' '}
        <code>MetalInfo</code> structs. It is given and not edited.
      </p>
      <p>
        <code>chooseLaunch</code> must return a block within <code>maxThreadsPerBlock</code> and{' '}
        <code>maxThreadsDim</code>, a whole number of warps by the reported <code>warpSize</code>, able to fill an SM's
        thread slots (neither so small that the block slots run out first nor a size that leaves slots over), no taller
        than the number of rows, and a grid that covers the output within <code>maxGridSize</code>, or return{' '}
        <code>false</code>. A field of 0 means the device did not report it, and the rule that needs it is skipped. The
        rules are this course's starting point for one thread per output, not a vendor's recommendation. On Metal the
        threadgroup follows Apple's recipe instead: the pipeline's <code>threadExecutionWidth</code> wide, as tall as{' '}
        <code>maxTotalThreadsPerThreadgroup</code> allows and the batch needs, dispatched with{' '}
        <code>dispatchThreads</code>.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md, A18, C2, C6)">
        <p>
          Do not read <code>devProp.clockRate</code>: CUDA 13.0 removed it, and HIP still has it, so one source that
          reads it builds on one and not the other. The per-SM register pool is <code>regsPerMultiprocessor</code>, not
          the <code>regsPerBlock</code> of p. 89. On Metal the width belongs to a compiled pipeline, not to the device.
        </p>
      </Today>
      <p>
        Run <code>python3 labs/pmpp.py run 4.1</code>, starting with <code>--backend emu</code>. Every field your{' '}
        <code>queryDevices</code> fills must equal what the harness reads on its own. <code>chooseLaunch</code> is run on
        nine synthetic devices, none of them a product, crossed with four output shapes, and each rule is checked on its
        own so a failure names the rule. The launch is compared with a CPU reference at 1 × 1000, 3 × 40, 37 × 53 and 64
        × 257 outputs. On Metal, <code>queryMetal</code> is compared on three pipelines, one of them compiled with a
        lower thread limit than the device's so that a value read from the wrong object shows; and the probe's records
        must satisfy what the specification promises about SIMD-groups, and nothing more.
      </p>
      <LabPrediction
        id="ch04/labs/p-lab1-speedup"
        prompt={
          <>
            Before you run the benchmark. It times a linear layer with 1024 inputs and 4096 outputs (the book's example
            layer of §20.6, which lab 5.3 also uses) at batch 1, once with your derived configuration and once with
            fixed 16 × 16 blocks. How many times as fast will yours be? (1 means the same speed, below 1 means slower.)
          </>
        }
        unit="× the speed of 16 × 16"
        measure={derivedOverFixed}
        missing={
          <>
            Once your lab passes, run <code>python3 labs/pmpp.py run 4.1 --bench</code> on a machine with a GPU, then
            reload.
          </>
        }
        reading={
          <>
            On CUDA and HIP a 16 × 16 block at batch 1 has 15 rows of threads with nothing to do. On Metal,{' '}
            <code>dispatchThreads</code> trims a threadgroup to the grid, so 16 × 16 runs as 16 × 1 there. Look at batch
            64 in the table before you decide what the number says about your rules. A configuration that loses is a
            finding about the rules, and the reason they live in one function.
          </>
        }
      />
      <LabPrediction
        id="ch04/labs/p-lab1-width"
        prompt={<>And the width your device will report: how many threads in a warp, or in a Metal SIMD-group?</>}
        unit="threads"
        measure={reportedWidth}
        missing={
          <>
            Run <code>python3 labs/pmpp.py run 4.1 --bench</code> on a machine with a GPU, then reload. The width is
            recorded with the benchmark.
          </>
        }
        reading={
          <>
            CUDA documents 32 on every compute capability it lists, HIP reports 64 or 32 depending on the AMD
            architecture, and Apple documents no value at all, only where to read it (notes/today/ch04-facts.md, A8, B6,
            C2). On Metal this is one pipeline's width; another kernel may report another.
          </>
        }
      />
      <LabResults
        lab={l1}
        filter={(r) => r.name === 'linear'}
        cols={[
          { head: 'Configuration', cell: (r) => (r.config === 'derived' ? 'derived (from the queries)' : 'fixed 16 × 16') },
          rawCol('Batch', 'batch'),
          { head: 'Block', num: true, cell: (r) => shape(r, 'block_x', 'block_y') },
          { head: 'Grid', num: true, cell: (r) => shape(r, 'grid_x', 'grid_y') },
          msCol,
          numCol('GFLOP/s', 'gflops'),
        ]}
      />
      <p>
        The same run records what your queries returned, which <a href={href('ch04', '4.8')}>section 4.8</a> sets out
        field by field, and on Metal what the probe found about SIMD-groups, which the warp painter of{' '}
        <a href={href('ch04', '4.4')}>section 4.4</a> draws.
      </p>
      <h3>Project milestone M1</h3>
      <p>
        Lab 3.3's kernel and this lab's launch together are milestone M1 of the through-line project: a linear layer
        whose launch is derived from what the device reports, with nothing about the device written into the source.
        Lab 5.3 replaces the kernel with a tiled one of the same arguments; the launch rules carry over.
      </p>
      <Beyond>
        <p>
          The through-line project, <code>nanoinfer</code>, is ours: one small GPU inference path for a decoder-only
          transformer, grown a milestone at a time.
        </p>
      </Beyond>
      <MilestoneStatus />
      <ReferenceSolution lab={l1} tiers={LINEAR_TIERS} />

      {/* ------------------------------------------------------------ 4.2 */}
      <h2>Lab 4.2: what a divergent warp costs</h2>
      <p>
        <a href={href('ch04', '4.5')}>Section 4.5</a> counts the passes a warp makes when its threads disagree at a
        branch. It gives no price. This lab measures one. You write three kernels: <code>branchKernel</code>, where
        each thread runs one of two equally long paths chosen by its element of <code>which</code> (the shape of Fig.
        4.9, with data in place of <code>threadIdx.x &lt; 24</code>); <code>selectKernel</code>, the same result with
        no thread's control flow depending on <code>which</code>; and <code>loopKernel</code>, where each thread loops
        its own number of times (the shape of Fig. 4.10). The driver then lays out the two paths in different
        patterns over the same data.
      </p>
      <Source section="4.5" pages="79–82" figs="Figs. 4.9 and 4.10; the kernels are ours" />
      <LabFiles
        lab={l2}
        files={[
          ['kernel.cu', <>the three kernels and <code>launchBranch</code>, <code>launchSelect</code>, <code>launchLoop</code></>],
          ['kernel.metal', <><code>branch_kernel</code>, <code>select_kernel</code> and <code>loop_kernel</code></>],
          ['launch.mm', <><code>encodeBranch</code>, <code>encodeSelect</code> and <code>encodeLoop</code></>],
        ]}
      />
      <p>
        The two paths are given in <code>paths.h</code> and <code>paths.metal.h</code>, beside your files and not to be
        edited. Each is a loop with data-dependent state, so that the branch is hard for a compiler to turn into a
        select behind your back; no vendor documents when a compiler does that, so this is our reasoning, and the
        benchmark is the evidence. The patterns are built from the width your device reports, never from a literal 32: on CUDA
        and HIP from <code>warpSize</code>, on Metal from what a provided probe kernel reports about which thread sits
        in which SIMD-group, because Apple leaves that cut to the implementation. The emulator does not run the threads of a
        warp in step, so it runs every pattern at pretend widths of 4, 32 and 64 to check your values.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md, A9, B7, C4)">
        <p>
          NVIDIA and AMD both document the passes: each path executed with the other threads masked off. Neither gives
          a cost. Apple documents neither the mechanism nor its cost, so on Metal your table is the only evidence.
        </p>
      </Today>
      <p>
        Run <code>python3 labs/pmpp.py run 4.2 --backend emu</code>, then <code>python3 labs/pmpp.py run 4.2</code>.
        The tests compare all three kernels with a CPU reference for eight patterns (all A, all B, the first half of the
        array A, alternating warps, half of each warp's lanes, alternating lanes, one lane per warp, a seeded random
        half), at block sizes of one, two and eight widths, with lengths that are and are not multiples of the block.{' '}
        <code>loopKernel</code> is tested with trip counts that include 0, sorted, shuffled and constant. A test can
        tell that your <code>selectKernel</code> computes the right values, not whether it avoids the branch: only the
        benchmark shows that.
      </p>
      <LabPrediction
        id="ch04/4.5/p-divcost"
        prompt={
          <>
            Before you run the benchmark (the same prediction as in section 4.5: commit it on either page). The same
            work, with the two paths mixed half and half inside every warp instead of sorted so that each warp is
            uniform. How many times as long will it take on your GPU?
          </>
        }
        unit="times as long"
        measure={divergenceCost}
        missing={
          <>
            Once your kernels pass, run <code>python3 labs/pmpp.py run 4.2 --bench</code> on a machine with a GPU, then
            reload.
          </>
        }
        reading={
          <>
            Section 4.5's pass count says two passes where a uniform warp makes one, for long, equal paths. A device
            can come out lower or higher than its pass count, and no vendor publishes a figure to compare with.
          </>
        }
      />
      <LabPrediction
        id="ch04/labs/p-lab2-select"
        prompt={
          <>
            On that same mixed pattern, computing both paths and selecting will take how many times as long as the
            branch? (Below 1 means the select is faster.)
          </>
        }
        unit="× the branch's time"
        measure={selectCost}
        missing={
          <>
            Run <code>python3 labs/pmpp.py run 4.2 --bench</code> on a machine with a GPU, then reload.
          </>
        }
        reading={
          <>
            The select always does both paths' work. The branch does both only in a warp that holds both. Compare the
            select's row on a uniform pattern in the table: that is its price where the branch costs nothing extra.
          </>
        }
      />
      <LabResults
        lab={l2}
        filter={(r) => r.name === 'branch' || r.name === 'select' || r.name === 'loop'}
        cols={[
          { head: 'Version', cell: (r) => r.name },
          { head: 'Pattern', cell: (r) => String(r.name === 'loop' ? `trip counts ${String(r.order)}` : r.pattern) },
          rawCol('Width', 'width'),
          msCol,
          numCol('÷ same mix, sorted', 'ratio_to_sorted'),
          numCol('Select ÷ branch', 'ratio_to_branch'),
        ]}
      />
      <ReferenceSolution lab={l2} tiers={DIVERGENCE_TIERS} />

      {/* ------------------------------------------------------------ 4.3 */}
      <h2>Lab 4.3: barrier hunt</h2>
      <p>
        You are handed a kernel that reverses each block's slice of an array in place: every thread reads its element
        into a register, then writes it to the mirrored position of its slice. It compiles, and it has three
        independent defects, the same three in the CUDA and the Metal version. Find and fix all three. They are the
        kinds an in-place exchange between threads invites: one about the order of reads and writes, one about the
        block size, one about the last, partial block.
      </p>
      <Source section="4.3" pages="72" figs="the any-order rule §4.4, p. 74; exercise 5, p. 91; the kernel is ours" />
      <LabFiles
        lab={l3}
        files={[
          ['kernel.cu', <><code>reverseBlocksKernel</code>, with the three defects; its launch is correct</>],
          ['kernel.metal', <><code>reverse_blocks</code>, with the same three</>],
          ['launch.mm', <>correct as shipped; nothing to change</>],
        ]}
      />
      <p>The method matters more than the fixes:</p>
      <ol>
        <li>
          Emulator first: <code>python3 labs/pmpp.py run 4.3 --backend emu</code>. It runs one thread until it reaches
          a barrier or returns, then the next, under three fixed orders, so a missing barrier is wrong every time. A
          write outside the array stops the run with an AddressSanitizer report: read <code>WRITE</code>, the first
          frame's <code>kernel.cu</code> line, and how many bytes past the region the address is.
        </li>
        <li>
          Fix one thing and run again. Which block sizes fail, and whether the length is a multiple of the block, is
          your evidence for the next defect.
        </li>
        <li>
          <code>python3 labs/pmpp.py run 4.3 --backend race</code> runs the threads concurrently under ThreadSanitizer
          and names the two source lines of an unsynchronised read and write.
        </li>
        <li>
          Then the GPU: <code>python3 labs/pmpp.py run 4.3</code>. The array sits between sentinel elements, each shape
          is launched 40 times on fresh data, and a failing case says in how many of the 40 it was wrong.
        </li>
      </ol>
      <p>
        The driver picks the block sizes: 32, 48, 64 and 96 threads, then one, two and four times the width your
        device reports, each with lengths that are and are not multiples of the block. The Metal barrier takes a flag
        that names the memory it orders; deciding which one belongs here is part of the lab.
      </p>
      <RaceStatus lab={l3} />
      <h3>The experiment of exercise 5</h3>
      <p>
        Exercise 5 claims that with 32 threads in a block the barrier can be left out. Once all your tests pass, delete
        the barrier again, run on the emulator (every case fails, under every order), then on your GPU five times,
        noting which block sizes fail and in how many of their 40 launches. Then put the barrier back.
      </p>
      <BarrierPrediction />
      <p>
        Whatever your GPU does, read it against what the vendors document, because that is what the next device is
        entitled to do. The width is not a constant. NVIDIA does not promise that the threads of a warp stay in step,
        AMD does, and Apple says nothing either way. And a block wider than one warp is several warps, which nothing
        but a barrier orders against each other.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md, A8, A9, A10, B5, B6, B8, C2)">
        <p>
          NVIDIA: from compute capability 7.0 each thread has its own program counter; synchronise warp-synchronous
          code explicitly. AMD: "all warp lanes are executed in lockstep". Apple: SIMD-groups "make independent
          forward progress" without a threadgroup barrier, and no width is documented.
        </p>
      </Today>
      <LabResults
        lab={l3}
        noBench="This lab has no benchmark. The result is the line above: every test passing on the emulators and on a device. As shipped, the kernel is meant to fail."
      />
      <ReferenceSolution lab={l3} tiers={BARRIER_TIERS} />

      {/* ------------------------------------------------------------ 4.4 */}
      <h2>Lab 4.4: block size on a kernel that waits on memory</h2>
      <p>
        You write a gather in the shape of an embedding bag: output <code>i</code> is the sum of <code>k</code> table
        entries chosen by an index array, <code>out[i] = Σ table[idx[i*k + j]]</code>. The indices are random and the
        table is large, so nearly every step is a load from global memory that nothing predicts. That is the kernel{' '}
        <a href={href('ch04', '4.6')}>section 4.6</a> is about: it spends its time waiting, and what it needs from the
        SM is other warps to run meanwhile. The block size is a run-time argument of the launch, and the benchmark
        sweeps it.
      </p>
      <Source section="4.7" pages="85–87" figs="latency tolerance §4.6, pp. 82–85; exercises 6 to 8, pp. 91–92; the kernel is ours" />
      <p>
        On CUDA, HIP and the emulator you also write the book's occupancy arithmetic as two functions,{' '}
        <code>modelBlocksPerSm</code> and <code>modelOccupancy</code>: block slots, thread slots counted in whole warps,
        and registers, with a limit of 0 meaning not reported. The benchmark sets your model beside the measurement and
        beside the vendor's own occupancy function.
      </p>
      <Beyond>
        <p>
          An embedding bag is ours as a bridge: it is the shape of the token-embedding lookup at the bottom of a model,
          and of any kernel whose loads nothing predicts.
        </p>
      </Beyond>
      <LabFiles
        lab={l4}
        files={[
          ['kernel.cu', <><code>gatherSumKernel</code>, <code>launchGatherSum</code>, <code>modelBlocksPerSm</code> and <code>modelOccupancy</code></>],
          ['kernel.metal', <><code>gather_sum</code></>],
          ['launch.mm', <><code>encodeGatherSum</code>, with the threadgroup size as an argument</>],
        ]}
      />
      <p>
        <code>sm_limits.h</code> beside them holds the limits struct and is given. The file{' '}
        <code>provided/occupancy_api.cu</code> is the driver's: on CUDA and HIP it asks the vendor's runtime how many
        blocks of your kernel fit on one SM. It has never been compiled by nvcc or a HIP compiler, so it is unverified on
        both; if it does not build with your toolchain, the lab's <code>README.md</code> says how to configure without it.
      </p>
      <p>
        Run <code>python3 labs/pmpp.py run 4.4</code>, starting with <code>--backend emu</code>. The kernel is compared
        with a CPU reference for several sizes and block sizes, most leaving a partial last block, and on the emulator
        AddressSanitizer stops at the first access outside a buffer. The model is tested on limits nobody has to own:
        the book's seven worked block sizes, 31 and 33 and 64 registers per thread, exercises 6, 7 and 8, a device with
        1536 thread slots and 16 block slots, blocks that are not whole warps, a width of 64, and limits of 0. In the
        results table, the columns for your model and the vendor's function stay empty on Metal and wherever the device
        left a limit unreported.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md, A13, B9, B10, C5)">
        <p>
          Registers are the model's weakest part: NVIDIA does not publish the granularity, AMD allocates per wavefront
          and its block-slot field is computed from the thread slots. Metal has no occupancy figure at all, so there the
          benchmark is a timing sweep and nothing else.
        </p>
      </Today>
      <LabPrediction
        id="ch04/4.7/p-blocksize"
        prompt={
          <>
            Before you run the benchmark (the same prediction as in section 4.7: commit it on either page). The sweep
            runs the same gather at 1, 2, 3, 4, 6, 8, 12, 16, 24 and 32 times the device's width, up to the largest
            block it accepts. How many times as long will the slowest block size take as the fastest? (1 means block
            size makes no difference.)
          </>
        }
        unit="times"
        measure={blockSizeSpread}
        missing={
          <>
            Once your kernel passes, run <code>python3 labs/pmpp.py run 4.4 --bench</code> on a machine with a GPU, then
            reload.
          </>
        }
        reading={
          <>
            A gap between two sizes no larger than the run-to-run spread in the same line is the machine, not the block
            size. A flat sweep is a result too: on a device where even small blocks fill the slots, block size has
            little left to change.
          </>
        }
      />
      <LabResults
        lab={l4}
        filter={(r) => r.name === 'sweep'}
        cols={[
          rawCol('Threads per block', 'threads_per_block'),
          rawCol('× width', 'multiple'),
          { ...msCol, head: 'Best, ms' },
          {
            head: 'Median, ms',
            num: true,
            cell: (r) => {
              const v = num(r.median_seconds);
              return v == null ? '' : fmt(v * 1000);
            },
          },
          {
            head: 'M lookups/s',
            num: true,
            cell: (r) => {
              const v = num(r.lookups_per_s);
              return v == null ? '' : fmt(v / 1e6, 4);
            },
          },
          {
            head: 'Your model',
            num: true,
            cell: (r) => {
              const v = num(r.model_occupancy);
              return v == null ? '' : `${fmt(v * 100)}%`;
            },
          },
          rawCol("Vendor's blocks per SM", 'vendor_blocks_per_sm'),
        ]}
      />
      <ReferenceSolution lab={l4} tiers={GATHER_TIERS} />
    </LabsPage>
  );
}
