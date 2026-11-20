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
  benchRow,
  defineLabs,
  measure,
  msCol,
  numCol,
  rawCol,
  useLabs,
  type BenchSet,
} from '../../components/LabKit';
import { benchRows, fmt, labStatus, num, realBackends, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 13's own content.
// No question on this page: the lab is the exercise. The two predictions are stored, not
// graded, under ch13/labs/p-coarsen and ch13/labs/p-tiled.
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch13', [{ num: '13.1', dir: 'lab1_merge', name: 'merge' }]);
const [l1] = LABS;

/** Identifier colours in the reference solution: where each value lives. */
const TIERS: Record<string, Tier> = {
  A: 'global', B: 'global', C: 'global', bounds: 'global', out: 'global',
  tile: 'shared', A_S: 'shared', B_S: 'shared', As: 'shared', Bs: 'shared',
  i: 'reg', j: 'reg', k: 'reg', i_low: 'reg', j_low: 'reg', delta: 'reg', active: 'reg',
  tid: 'reg', k_curr: 'reg', k_next: 'reg', i_curr: 'reg', i_next: 'reg', j_curr: 'reg', j_next: 'reg',
  b: 'reg', tileSize: 'reg', numTiles: 'reg', C_curr: 'reg', C_next: 'reg', A_curr: 'reg', A_next: 'reg',
  B_curr: 'reg', B_next: 'reg', A_length: 'reg', B_length: 'reg', C_length: 'reg', c_curr: 'reg', c_next: 'reg',
  a_curr: 'reg', a_next: 'reg', b_curr: 'reg', b_next: 'reg', iTooHigh: 'reg', jTooHigh: 'reg',
};

// ---------------------------------------------------------------- what each prediction is measured against
const ms = (r: BenchRow | undefined) => {
  const v = num(r?.seconds);
  return v ? `${fmt(v * 1000)} ms` : '?';
};
const size = (r: BenchRow | undefined) => (num(r?.total) ? `${num(r?.total)!.toLocaleString('en-US')} outputs` : '');

/** Basic kernel: time at 1 output per thread over time at 8. Above 1, coarsening was faster. */
const coarsen = (reports: Reports) =>
  measure(reports, l1, (s: BenchSet) => {
    const one = benchRow(s, 'basic', { per_thread: 1 });
    const eight = benchRow(s, 'basic', { per_thread: 8 });
    const a = num(one?.seconds);
    const b = num(eight?.seconds);
    if (!a || !b) return null;
    return { value: a / b, detail: `${ms(eight)} at 8 outputs per thread against ${ms(one)} at 1, ${size(eight)}` };
  });

/** At 8 outputs per thread: basic time over tiled time (the tiled time includes its partition launch). */
const tiled = (reports: Reports) =>
  measure(reports, l1, (s: BenchSet) => {
    const basic = benchRow(s, 'basic', { per_thread: 8 });
    const tile = benchRow(s, 'tiled', { per_thread: 8 });
    const a = num(basic?.seconds);
    const b = num(tile?.seconds);
    if (!a || !b) return null;
    return { value: a / b, detail: `${ms(tile)} tiled, partition launch included, against ${ms(basic)} basic, ${size(tile)}` };
  });

const MISSING = (
  <>
    This needs the benchmark rows of lab 13.1 from a real device. Run <code>python3 labs/pmpp.py run 13.1 --bench</code>{' '}
    once your kernels pass (or with <code>--solution</code> to time the reference). The emulator is never benchmarked.
  </>
);

// ---------------------------------------------------------------- results table
const kernelName = (r: BenchRow): string =>
  r.name === 'basic' ? 'basic, searches global memory' : r.name === 'tiled' ? 'tiled, through shared memory' : r.name;

/** One line on lab 13.1 on the race backend, which the status table does not show. */
function RaceStatus() {
  const { reports, loading } = useLabs();
  const run = reports.race;
  const line = (variant: 'solution' | 'work') => {
    const rs = (run?.results.filter((x) => x.lab === l1.id && x.variant === variant) ?? []) as (RunResult & { tsan?: unknown[] })[];
    const tests = rs.flatMap((x) => x.tests);
    const reportsFound = rs.reduce((n, x) => n + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0);
    return { s: labStatus(run, l1.id, variant), tests, passed: tests.filter((t) => t.pass).length, reportsFound };
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
              ? `On this machine the reference solution passed all ${sol.tests.length} tests on the race backend with ${sol.reportsFound ? `${sol.reportsFound} ThreadSanitizer report(s)` : 'no race reported'}: emulator-checked on this flavour.`
              : `On this machine the reference solution failed on the race backend (${sol.passed} of ${sol.tests.length} tests): unverified there.`}{' '}
          {mine.s === 'none'
            ? 'Your own kernel has no recorded run there, or is still the untouched starter.'
            : mine.s === 'pass'
              ? `Your own kernel passed all ${mine.tests.length} tests there with ${mine.reportsFound ? `${mine.reportsFound} ThreadSanitizer report(s): read them` : 'no race reported'}.`
              : `Your own kernel passed ${mine.passed} of ${mine.tests.length} tests there, with ${mine.reportsFound} ThreadSanitizer report(s) in the run.`}
        </>
      )}
    </p>
  );
}

/** The rows a backend's comparison uses: the reader's own where they passed, else the reference's. */
function preferredRows(reports: Reports, b: string, lab: string) {
  const run = reports[b];
  const mine = labStatus(run, lab, 'work') === 'pass' ? benchRows(run, lab, 'work') : [];
  return mine.length ? mine : benchRows(run, lab, 'solution');
}

/** The tiled kernel's minimum-traffic rate beside lab 5.1's measured bandwidth, where both exist. */
function AgainstBandwidth() {
  const { reports, loading } = useLabs();
  if (loading) return null;
  const lines = realBackends(reports).flatMap((b) => {
    const rows = preferredRows(reports, b, l1.id).filter((r) => r.name === 'tiled' && num(r.gb_per_s));
    const best = rows.reduce<BenchRow | undefined>((x, r) => (!x || num(r.gb_per_s)! > num(x.gb_per_s)! ? r : x), undefined);
    const roof = preferredRows(reports, b, 'ch05/lab1_roofline').find((r) => r.name === 'roofline');
    const bw = num(roof?.bandwidth_gb_per_s);
    const g = num(best?.gb_per_s);
    if (!best || !g || !bw) return [];
    return [{ b, device: String(reports[b].device.name ?? b), g, bw, per: best.per_thread }];
  });
  return lines.length ? (
    <>
      {lines.map((l) => (
        <p key={l.b} data-bandwidth={l.b} data-share={fmt(l.g / l.bw)}>
          On {l.device} ({l.b}) the tiled kernel's best row, at {String(l.per)} outputs per thread, moved data at{' '}
          <b>{fmt(l.g)} GB/s</b> counted at the minimum traffic, and lab 5.1's bandwidth probe measured{' '}
          <b>{fmt(l.bw)} GB/s</b>: a ratio of {fmt(l.g / l.bw)}.
        </p>
      ))}
    </>
  ) : (
    <p className="muted" data-bandwidth="none">
      To set the merge beside your device's bandwidth you need benchmark rows from lab 13.1 and from lab 5.1 on the same
      backend: <code>python3 labs/pmpp.py run 13.1 --bench</code> and <code>python3 labs/pmpp.py run 5.1 --bench</code>.
    </p>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS} tiers={TIERS}>
      <LabsIntro>
        One lab, in <code>labs/ch13/lab1_merge/</code>. It is optional, as this chapter is condensed, and it adds nothing
        to the through-line project. It builds the chapter's two merge kernels, the second in the shape the libraries
        ship, and it tests stability, which a merge of plain integers cannot show.
      </LabsIntro>
      <RunningLabs
        example="13.1"
        alsoShownIn={
          <>
            the tiled kernel's listing on <a href={href('ch13', '13.3')}>13.3</a>
          </>
        }
      />
      <BackendsAndEmulator />
      <VerificationStatus />

      {/* ------------------------------------------------------------ 13.1 */}
      <h2>Lab 13.1: merge two sorted arrays with co-rank partitioning</h2>
      <p>
        A and B are sorted arrays of key–value pairs, <code>struct KV {'{'} int key; int val; {'}'}</code>, and you write the
        first <code>total</code> outputs of their stable merge to C, comparing keys only. Ties go to A, and elements from
        the same array keep their order: the book's rule. <code>total = m + n</code> is the whole merge. You write it
        twice, in CUDA and in Metal.
      </p>
      <Source section="13.2 to 13.6" pages="305–320" figs="Figs. 13.2, 13.5, 13.9; Exercise 3" />
      <p>
        Part (a) is the basic kernel of page <a href={href('ch13', '13.3')}>13.3</a>, corrected: <code>co_rank</code>,{' '}
        <code>merge_sequential</code> with <code>&lt;=</code>, and <code>mergeBasicKernel</code>, where each thread owns{' '}
        <code>perThread</code> consecutive outputs, limits its first and the next thread's first output to{' '}
        <code>total</code>, returns if it owns none, and then makes two <code>co_rank</code> calls and one merge. Its
        launcher uses blocks of 128 threads (<code>MERGE_BLOCK</code>), enough of them for <code>total</code> outputs by
        integer ceiling division, and launches nothing when <code>total</code> is 0.
      </p>
      <p>
        Part (b) stages the inputs in shared memory with exact loads, the design of page 13.3's "Load exactly what a tile
        uses". <code>partitionKernel</code> stores the co-rank of every tile boundary in <code>bounds</code>, a tile being
        the <code>128 × perThread</code> outputs of one block. <code>mergeTiledKernel</code> gives block b the range{' '}
        <code>A[bounds[b] .. bounds[b+1])</code> and the B range that makes up the rest of its outputs, copies both into
        one <code>extern __shared__ KV tile[]</code> with consecutive threads on consecutive elements, waits at one{' '}
        <code>__syncthreads()</code>, and then each thread searches the shared copies and merges into C. The launcher
        makes the two launches, giving the second <code>tile × sizeof(KV)</code> bytes of dynamic shared memory.
      </p>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <>
              <code>co_rank</code>, <code>merge_sequential</code>, <code>mergeBasicKernel</code>,{' '}
              <code>partitionKernel</code>, <code>mergeTiledKernel</code>, <code>launchMergeBasic</code> and{' '}
              <code>launchMergeTiled</code>; <code>coRankKernel</code> and <code>launchCoRank</code> are provided so the
              tests can call your <code>co_rank</code> on its own
            </>,
          ],
          [
            'kernel.metal',
            <>
              <code>co_rank</code>, <code>merge_sequential</code>, <code>merge_basic</code>, <code>merge_partition</code>{' '}
              and <code>merge_tiled</code>; <code>co_rank_test</code> is provided
            </>,
          ],
          [
            'launch.mm',
            <>
              <code>encodeMergeBasic</code> and <code>encodeMergeTiled</code>; <code>encodeCoRank</code> is provided
            </>,
          ],
        ]}
      />
      <p>
        <code>merge.h</code> (the struct, the block size and the size of <code>bounds</code>) and <code>main.cpp</code>{' '}
        are read-only.
      </p>
      <p>
        A <code>total</code> below m + n is ours, not the book's. It is the merge of two top-k lists, where only the
        first k outputs are wanted: a thread whose first output is at or past <code>total</code> has nothing to do. That
        is how llama.cpp's Metal backend combines per-block top-k lists, and a later sampling module of this course can
        reuse it.
      </p>
      <Beyond>
        <p>
          The <code>total</code> mode and the top-k use are ours; the llama.cpp kernel is in notes/today/ch13-facts.md,
          D3. A top-k list is sorted largest first, so it merges with the keys negated or every comparison reversed.
        </p>
      </Beyond>
      <p>
        In Metal two things change. The search and the merge run on device memory in <code>merge_basic</code> and on
        threadgroup memory in <code>merge_tiled</code>, and since every Metal pointer names one address space, the single{' '}
        <code>const KV*</code> helper of the CUDA file cannot serve both. The starter makes each helper a template on the
        pointer type, as PyTorch's MPS merge sort does; MLX writes two overloads instead. And because the Metal
        specification does not restate C++'s rule that <code>&amp;&amp;</code> skips its right operand, test the bounds
        first and read inside the <code>if</code>. <code>encodeMergeTiled</code> puts the partition and the merge in one
        encoder, which runs them in order, and its threadgroup memory, <code>tile × 8</code> bytes, is always the multiple
        of 16 that Metal requires.
      </p>
      <Today checked="2026-11-20 (notes/today/ch13-facts.md C2, C3; ch02-facts C9; ch05-facts C8)">
        <p>
          MSL address spaces are disjoint, and overloading extends to them. The spec says nothing of the order in which{' '}
          <code>&amp;&amp;</code> evaluates. Two dispatches in one default compute encoder run in order.
        </p>
      </Today>

      <h3>Where the lab differs from the page</h3>
      <ul>
        <li>
          Integer ceiling division. Fig. 13.9 line 03 takes <code>ceil</code> of an integer quotient, which has already
          been truncated. Here the per-thread count is an argument and the launcher sizes the grid with{' '}
          <code>(total + per·128 − 1) / (per·128)</code>.
        </li>
        <li>
          <code>__device__</code> on <code>co_rank</code> and <code>merge_sequential</code>: the listings show none, and on
          CUDA and HIP an unqualified function is host-only.
        </li>
        <li>
          A thread whose first output is at or past <code>total</code> returns. The book clamps only{' '}
          <code>k_next</code> and relies on the launch.
        </li>
        <li>Key–value pairs instead of <code>int</code>, so that stability can be tested, and <code>const</code> inputs.</li>
        <li>
          Exact loading, not §13.6's full tile of each input: every block's co-ranks are computed first, in their own
          launch, where Fig. 13.11 has thread 0 compute them between two barriers. One tile per block, so no while-loop,
          no <code>A_consumed</code>, and nothing left over for a circular buffer to keep.
        </li>
      </ul>

      <h3>What the tests check</h3>
      <p>
        Run <code>python3 labs/pmpp.py run 13.1 --backend emu</code> first. Every check is exact equality: a merge copies
        elements and does no arithmetic, so no tolerance applies. First <code>co_rank</code> alone, through the provided
        kernel: on the running example at every k from 0 to 9, then on random arrays with many equal keys at every k,
        each expected value counted from <code>std::merge</code>. Then both merges on empty inputs, one empty input,
        all keys equal, one array entirely below the other, very unequal lengths and random inputs, at 1, 3 and 8 outputs
        per thread (devices add 16 and two large sizes), and at several values of <code>total</code>.
      </p>
      <p>
        Element <code>A[i]</code> carries the value i and <code>B[j]</code> the value 1,000,000,000 + j, so every output
        says where it came from. The reference is <code>std::merge</code> on keys only, which the C++ standard guarantees
        to be stable with the first range first, so an exact match of <code>C[0 .. total)</code> also proves stability.
        A failure reads like <i>C[6] came from B[1]; a stable merge takes A[4] first (both have key 10)</i>. C starts full
        of a sentinel, so an output you never wrote is reported, and so is anything written at or past{' '}
        <code>C[total]</code>.
      </p>
      <Today checked="2026-11-20 (notes/today/ch13-facts.md A6)">
        <p>
          The C++ draft: <code>std::merge</code> is "Stable", and for equivalent elements those from the first range
          precede those from the second.
        </p>
      </Today>
      <p>
        The tiled kernel shares its tile between threads behind one barrier, so the lab is also built for the emulator's
        race flavour: <code>python3 labs/pmpp.py run 13.1 --backend race</code> runs the tiled kernel's threads
        concurrently under ThreadSanitizer, on the running example and on random arrays of 300 and 257 elements at 4 and
        2 outputs per thread. With the barrier removed from the reference solution, it reported the tile's stores
        against the searches' and merges' reads of it. A clean run is evidence, not proof: it sees only the
        schedules that happened.
      </p>
      <RaceStatus />
      <p>
        Do not return early from <code>mergeTiledKernel</code>. Every thread must reach the barrier, including a thread
        of the last block that owns no outputs: it may still have elements to load. Clamp its output range to the
        block's length instead, so that it merges zero elements.
      </p>

      <h3>The experiment</h3>
      <p>
        Do this once your kernels pass, and before you run <code>--bench</code>. Page 13.3's warp explorer says that at
        one output per thread every output pays for two searches but the stores coalesce, and at eight the searches are
        shared while each lane stores into its own segment. It also says the tiled kernel moves the irregular reads into
        shared memory. Both are counts per instruction, with no caches. Write down two numbers.
      </p>
      <Source section="13.5, 13.6, 13.8" pages="314, 320, 327" />
      <LabPrediction
        id="ch13/labs/p-coarsen"
        prompt={<>For the basic kernel, how many times faster will 8 outputs per thread run than 1 on your GPU? (Below 1 means slower.)</>}
        unit="× the speed at 1 output per thread"
        measure={coarsen}
        missing={MISSING}
        reading={
          <>
            The model makes no promise either way: fewer searches pull one way, scattered stores the other. Each thread
            of the basic kernel walks its own slice in order, so a cache may serve its later steps from lines its earlier
            steps fetched, which the model leaves out.
          </>
        }
      />
      <LabPrediction
        id="ch13/labs/p-tiled"
        prompt={<>At 8 outputs per thread, how many times faster than the basic kernel will the tiled kernel run, partition launch included?</>}
        unit="× the basic kernel"
        measure={tiled}
        missing={MISSING}
        reading={
          <>
            The tiled kernel adds a launch and a barrier, and still searches twice per thread, in shared memory. What it
            removes is the scattered global traffic, so the ratio says how much of that traffic was really reaching
            memory on your device.
          </>
        }
      />
      <LabResults
        lab={l1}
        cols={[
          { head: 'What ran', cell: kernelName },
          rawCol('Outputs per thread', 'per_thread'),
          { head: 'Outputs', num: true, cell: (r) => (num(r.total) ? num(r.total)!.toLocaleString('en-US') : '') },
          msCol,
          numCol('GB/s at minimum traffic', 'gb_per_s'),
        ]}
      />
      <p>
        How the benchmark works: keys are uniform random 31-bit integers, sorted, with m = n. It grows the inputs from
        2^20 elements each until the basic kernel at 8 outputs per thread takes at least 20 ms, or each input reaches
        2^23 elements, with <code>total = m + n</code>. Each row is the best of several runs, and every result is checked
        against the CPU. "GB/s at minimum traffic" is 2 × total × 8 bytes over the time: every element read once and
        written once, the least any merge can move, not what the kernel actually moved. A row the device cannot run is
        printed as skipped, with the reason.
      </p>
      <AgainstBandwidth />
      <ReferenceSolution lab={l1} />
    </LabsPage>
  );
}
