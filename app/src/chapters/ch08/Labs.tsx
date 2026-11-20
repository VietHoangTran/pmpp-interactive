import type { Question } from '../../components/Ask';
import { Aside, Source, Today, type Tier } from '../../components/notes';
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
  preferredBench,
  rawCol,
  useLabs,
  type BenchSet,
} from '../../components/LabKit';
import { fmt, labStatus, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 8's own content.
// No question on this page: the lab is the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch08', [{ num: '8.1', dir: 'lab1_stencil3d', name: '3D stencil' }]);
const [l1] = LABS;

/** Identifier colours in the reference solution: where each value lives. */
const TIERS: Record<string, Tier> = {
  in: 'global',
  out: 'global',
  inCurr_s: 'shared',
  inPrev: 'reg',
  inCurr: 'reg',
  inNext: 'reg',
  iStart: 'reg',
  i: 'reg',
  j: 'reg',
  k: 'reg',
};

// ---------------------------------------------------------------- what each prediction is measured against
/** The book's configuration: 32 × 32 blocks, a slab of 30 output planes. */
const BOOK_TILE = 32;
const BOOK_DEPTH = BOOK_TILE - 2;
const bookRegtile = (s: BenchSet) => benchRow(s, 'regtile', { tile: BOOK_TILE, depth: BOOK_DEPTH });
const ms = (r: BenchRow | undefined) => {
  const v = num(r?.seconds);
  return v ? `${fmt(v * 1000)} ms` : '?';
};

/** Register-tiled at the book's slab over the basic kernel. */
const overBasic = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const r = bookRegtile(s);
    const v = num(r?.speedup_vs_basic);
    if (!r || !v) return null;
    return { value: v, detail: `${ms(r)} against ${ms(benchRow(s, 'basic'))} for the basic kernel, grid side ${String(r.n)}` };
  });

/** Register-tiled at the book's slab over the provided three-shared-plane kernel, same tile. */
const overPlanes = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const r = bookRegtile(s);
    const v = num(r?.speedup_vs_planes);
    if (!r || !v) return null;
    return { value: v, detail: `${ms(r)} against ${ms(benchRow(s, 'planes', { tile: BOOK_TILE }))} for the planes kernel, grid side ${String(r.n)}` };
  });

/** The slab depth whose register-tiled run at T = 32 was fastest. */
const fastestDepth = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const rows = s.rows
      .filter((r) => r.name === 'regtile' && num(r.tile) === BOOK_TILE && num(r.seconds) && num(r.depth))
      .sort((a, b) => num(a.depth)! - num(b.depth)!);
    if (rows.length < 2) return null;
    const best = rows.reduce((a, b) => (num(b.seconds)! < num(a.seconds)! ? b : a));
    return {
      value: num(best.depth)!,
      detail: rows.map((r) => `depth ${String(r.depth)}${r.depth === r.n ? ' (one slab)' : ''}: ${ms(r)}`).join(', '),
    };
  });

const MISSING = (
  <>
    This needs the register-tiled kernel's row at T = 32 and the book's depth of 30. Run{' '}
    <code>python3 labs/pmpp.py run 8.1 --bench</code> on a machine with a GPU once your kernel passes (or with{' '}
    <code>--solution</code> to time the reference). On Metal, T = 32 is skipped where the pipeline allows fewer than
    1024 threads; the limits below say whether that happened.
  </>
);

// ---------------------------------------------------------------- results table
const variant = (r: BenchRow): string =>
  r.name === 'basic'
    ? 'basic, 8 × 8 × 8 blocks'
    : r.name === 'cube'
      ? `cube, T = ${String(r.tile)} (provided)`
      : r.name === 'planes'
        ? `three shared planes, T = ${String(r.tile)} (provided)`
        : r.name === 'regtile'
          ? `register-tiled, T = ${String(r.tile)}`
          : r.name;
const depthCell = (r: BenchRow): string => {
  const d = num(r.depth);
  if (d == null) return '';
  if (d === num(r.n)) return `${d} (one slab)`;
  if (d === (num(r.tile) ?? 0) - 2) return `${d} (book's)`;
  return String(d);
};

/** One line on lab 8.1 on the race backend, which the status table does not show. */
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

/** The thread limits lab 8.1 recorded: on Metal each belongs to a compiled pipeline. */
function Limits() {
  const { reports, loading } = useLabs();
  const sets = preferredBench(reports, l1).filter((s) => s.rows.some((r) => r.name === 'limits'));
  if (loading) return <p className="muted">Reading reports.</p>;
  if (!sets.length)
    return (
      <p className="muted" data-limits="none">
        No limits recorded on this machine yet. The benchmark writes them, one per kernel and tile width.
      </p>
    );
  return (
    <ul data-limits="list">
      {sets.map((s) => {
        const rows = s.rows.filter((r) => r.name === 'limits');
        const cube = rows.find((r) => r.kernel === 'cube');
        const reg = rows.filter((r) => r.kernel !== 'cube').sort((a, b) => (num(a.tile) ?? 0) - (num(b.tile) ?? 0));
        const parts = [
          cube ? `cube kernel (512 threads): ${String(cube.max_threads)}` : '',
          ...reg.map((r) => `register-tiled at T = ${String(r.tile)} (${(num(r.tile) ?? 0) ** 2} threads): ${String(r.max_threads)}`),
        ].filter(Boolean);
        return (
          <li key={s.backend} data-limits={s.backend}>
            {s.backend} ({s.device}), run of {s.date}: {parts.join('; ')}.
          </li>
        );
      })}
    </ul>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS} tiers={TIERS}>
      <LabsIntro>
        One lab, in <code>labs/ch08/lab1_stencil3d/</code>. It is optional, as this chapter is condensed, and it adds
        nothing to the through-line project: it builds the first and the last kernel of the chapter and measures the
        model that <a href={href('ch08', '8.4')}>8.4</a> and <a href={href('ch08', '8.5')}>8.5</a> argue from.
      </LabsIntro>
      <RunningLabs
        example="8.1"
        alsoShownIn={
          <>
            the thread-limit options of the explorers on <a href={href('ch08', '8.4')}>8.4</a> and{' '}
            <a href={href('ch08', '8.5')}>8.5</a>
          </>
        }
      />
      <BackendsAndEmulator />
      <VerificationStatus />

      {/* ------------------------------------------------------------ 8.1 */}
      <h2>Lab 8.1: one thread per point, then marching in z with registers</h2>
      <p>
        A 7-point stencil sweep over an N × N × N grid of floats, row-major with x fastest. Every interior point of{' '}
        <code>out</code> becomes a weighted sum of the same point of <code>in</code> and its six axis neighbours; the
        boundary layer keeps its input values. You write two kernels, in CUDA and in Metal. The first is the basic kernel
        of Fig. 8.6: one thread per point, every operand from global memory, 8 × 8 × 8 blocks as in exercise 1b. The
        second is Fig. 8.12, corrected: a block of <code>TILE_WIDTH</code> × <code>TILE_WIDTH</code> threads, one x-y
        plane of the input tile, marches through a slab of output planes, keeps each thread's previous, current and next
        values in registers, and mirrors only the current plane in shared memory.
      </p>
      <Source section="8.2 to 8.6" pages="187–198" figs="Figs. 8.6, 8.12; exercise 1b" />
      <p>
        Two comparison kernels are provided, corrected and tested by the same driver: the cube-tiled kernel of Fig. 8.8
        (8 × 8 × 8 blocks) and the three-shared-plane kernel of Fig. 8.10, in <code>labs/ch08/shared/</code>. You do
        not edit them; <a href={href('ch08', '8.4')}>8.4</a> and <a href={href('ch08', '8.5')}>8.5</a> show them. With
        them the benchmark times all four of the chapter's kernels on one grid.
      </p>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <>
              <code>stencilBasicKernel</code>, <code>stencilRegTiledKernel</code> and their launches,{' '}
              <code>launchStencilBasic</code> and <code>launchStencilRegTiled</code>
            </>,
          ],
          [
            'kernel.metal',
            <>
              <code>stencil_basic</code> and <code>stencil_regtile</code>; every position attribute is a{' '}
              <code>uint3</code>
            </>,
          ],
          [
            'launch.mm',
            <>
              the two dispatches, in <code>encodeStencilBasic</code> and <code>encodeStencilRegTiled</code>
            </>,
          ],
        ]}
      />
      <p>
        The slab depth is a run-time argument, <code>depth</code>, where the book fixes it at{' '}
        <code>OUT_TILE_DIM</code>: with <code>depth = TILE_WIDTH − 2</code> your kernel is the book's, and the benchmark
        can try others. The grid is <code>ceil(N / (TILE_WIDTH − 2))</code> blocks in x and y and{' '}
        <code>ceil(N / depth)</code> in z, and slab z starts at plane <code>z · depth</code>. The build defines{' '}
        <code>TILE_WIDTH</code>: 32 by default, and 16 and 8 as well for the benchmark, so nothing may assume 32.
      </p>
      <p>Three places where the lab differs from the printed listings, each tested:</p>
      <ul>
        <li>
          <code>c3</code> multiplies the y − 1 neighbour and <code>c4</code> the y + 1 neighbour, as in Figs. 8.6 and
          8.8. Figs. 8.10 and 8.12 print the pair the other way round. The tests use seven different weights, so the
          printed pairing fails.
        </li>
        <li>
          The last line of Fig. 8.12's loop stores <code>inNext_s</code>, which that kernel does not have. The value to
          store is the register <code>inNext</code>.
        </li>
        <li>
          The listings use <code>c0</code> to <code>c6</code> without declaring them. Here they arrive as one
          by-value struct, <code>StencilCoeffs</code> in <code>stencil.h</code>.
        </li>
      </ul>
      <p>
        Run <code>python3 labs/pmpp.py run 8.1</code>, starting with <code>--backend emu</code>. Before every launch the
        driver copies <code>in</code> into <code>out</code> and writes a marker after the end of <code>out</code>, then
        compares every point with a CPU reference within a tolerance scaled to the size of its seven terms. So an
        unwritten interior point, a changed boundary point and a write past the end are each reported by name. On the
        emulator the grid sides are 3, 4, 5, 33, 34 and 63, at tile width T = 32, with slab depths of 1, 7, T − 2 and the
        whole column, under three thread orders; it takes a few minutes, smallest grids first. On a device the sides are 3,
        34, 65, 130 and 257 with the same depths, and Metal also runs tile widths 8 and 16. The provided kernels are
        tested by the same cases.
      </p>
      <p>
        Your kernel shares <code>inCurr_s</code> between threads and guards it with two barriers, so the lab is also
        built for the emulator's race flavour: <code>python3 labs/pmpp.py run 8.1 --backend race</code> runs the threads
        of a block concurrently under ThreadSanitizer, on grid sides 3, 4, 5 and 34 at T = 32. Where the plain emulator
        shows a missing barrier only as wrong values, the race flavour names the two lines that are not ordered: with
        barrier 2 removed from the reference solution it reported the store of the new current plane into{' '}
        <code>inCurr_s</code> against the neighbours' reads of it. A clean run is evidence, not proof: it sees only the
        schedules that happened.
      </p>
      <RaceStatus />
      <p>
        Do not return early. Every thread of the block must reach both barriers in every iteration of the z loop: a
        thread on the block's edge or the grid's boundary computes nothing but still loads its element of the halo.
        Put the conditions around the loads and the store, never a <code>return</code> ahead of the loop's barriers. The
        emulator cannot catch an early return by itself; it catches the damage, a halo value nobody loaded.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A6, B3, C3)">
        <p>
          NVIDIA's barrier waits for all non-exited threads, and Apple says the same for Apple silicon; the emulator
          models that. AMD states no rule. The portable rule is the book's: every thread reaches every barrier.
        </p>
      </Today>
      <p>
        One trap the tests cannot show here. On CUDA a 1024-thread block leaves at most 64 registers per thread; a kernel
        compiled to more fails at launch with "too many resources requested". If your T = 32 build does that,{' '}
        <code>nvcc --resource-usage</code> shows the count, and <a href={href('ch08', '8.5')}>8.5</a> explains the fix,{' '}
        <code>__launch_bounds__</code>. The lab's kernels do not carry it yet, for a reason of our own tooling: the
        emulator and the stub headers of the CUDA front-end check do not define it, so a kernel that used it would not
        build there. On Metal the driver reads each pipeline's thread cap and skips T = 32, with a message, where it is
        below 1024.
      </p>
      <Today checked="2026-11-20 (notes/today/ch08-facts.md A4, B4, C1)">
        <p>
          NVIDIA documents <code>__launch_bounds__</code> as the fix for that launch failure. HIP compiles for the
          device's largest block unless told otherwise, so it limits registers, and may spill, instead of failing. Metal
          may lower a pipeline's thread cap for a kernel that needs more resources.
        </p>
      </Today>

      <h3>The experiment</h3>
      <p>
        Do this once your kernels pass. The model says the basic kernel asks for 32 bytes per output point, 0.41 FLOP/B,
        and the register-tiled kernel at the book's slab, T = 32 and 30 planes, loads 32 × 32 × 32 values and stores 30 ×
        30 × 30 per block: 1.47 FLOP/B. That is about 3.6 times fewer bytes requested. Write down three predictions
        before you run <code>--bench</code>.
      </p>
      <Source section="8.3, 8.5" pages="189, 195" />
      <LabPrediction
        id="ch08/labs/p-lab1-speedup"
        prompt={
          <>
            At T = 32 and the book's slab depth of 30, how many times faster than the basic kernel will your
            register-tiled kernel run on your GPU?
          </>
        }
        unit="× the basic kernel"
        measure={overBasic}
        missing={MISSING}
        reading={
          <>
            3.6 is a ratio of requested bytes. Caches serve some of the basic kernel's neighbour reads (the book says
            so on p. 189), and the measured factor is smaller wherever they were already doing that work. The "GB/s at
            8 B/point" column below shows how close each kernel comes to moving only the compulsory bytes.
          </>
        }
      />
      <LabPrediction
        id="ch08/labs/p-lab1-planes"
        prompt={
          <>
            The provided three-shared-plane kernel moves exactly the same global data as yours (p. 198); yours swaps two
            shared planes for registers. How many times faster than it will yours run, at the same T = 32 and depth?
          </>
        }
        unit="× the planes kernel"
        measure={overPlanes}
        missing={MISSING}
        reading={
          <>
            Same global traffic, so any difference is on chip: fewer shared-memory reads and no plane copies, against
            one more write of the current value. No vendor documents how much a register access saves over a shared
            one (<a href={href('ch08', '8.5')}>8.5</a>, "What vendors document about registers").
          </>
        }
      />
      <LabPrediction
        id="ch08/labs/p-lab1-depth"
        prompt={
          <>
            At T = 32 the benchmark tries slab depths of 8, 16, 30 (the book's), 64 and the whole column. Which depth, in
            planes, will be fastest on your GPU?
          </>
        }
        unit="planes"
        measure={fastestDepth}
        missing={MISSING}
        reading={
          <>
            The model's ratio rises with depth towards 1.52, while the grid has fewer blocks and each does more work in
            sequence (<a href={href('ch08', '8.5')}>8.5</a>, "The numbers"). Neighbouring depths can differ by a few
            per cent from one run to the next; run the benchmark twice before you read much into a small gap.
          </>
        }
      />
      <LabResults
        lab={l1}
        filter={(r) => r.name !== 'limits'}
        cols={[
          { head: 'What ran', cell: variant },
          { head: 'Slab depth', num: true, cell: depthCell },
          rawCol('N', 'n'),
          msCol,
          numCol('Gpoints/s', 'gpts'),
          numCol('GB/s at 8 B/point', 'gbps_compulsory'),
          numCol('× basic', 'speedup_vs_basic'),
          numCol('× planes', 'speedup_vs_planes'),
        ]}
      />
      <p>
        How the benchmark works: it grows N until the basic kernel takes long enough to time, without letting{' '}
        <code>in</code> and <code>out</code> together pass 2 GiB or an eighth of the memory the device reports, then
        times every kernel on that grid, best of several runs, and checks each result against the CPU afterwards. A
        kernel whose output is wrong is reported, not timed. "GB/s at 8 B/point" counts one 4-byte load and one 4-byte
        store per interior point, the traffic of the 13/8 ideal, not what the kernel asked for.
      </p>
      <p>The thread limits the benchmark read, per kernel and tile width, in threads per block:</p>
      <Aside>
        <p>
          On Metal the limit belongs to a compiled pipeline (<code>maxTotalThreadsPerThreadgroup</code>), so each kernel
          has its own; on CUDA and HIP it is the device's <code>maxThreadsPerBlock</code>.
        </p>
      </Aside>
      <Limits />
      <ReferenceSolution lab={l1} />
    </LabsPage>
  );
}
