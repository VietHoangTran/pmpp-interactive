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
  preferredBench,
  rawCol,
  useLabs,
  type BenchSet,
  type Col,
  type Lab,
} from '../../components/LabKit';
import { fmt, labStatus, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 15's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

export const LABS = defineLabs('ch15', [
  { num: '15.1', dir: 'lab1_coarse', name: 'coarsened' },
  { num: '15.2', dir: 'lab2_regtile', name: 'register tiling' },
  { num: '15.3', dir: 'lab3_remap', name: 'coalesced stores' },
  { num: '15.4', dir: 'lab4_padded', name: 'padding, vector loads' },
  { num: '15.5', dir: 'lab5_pipelined', name: 'pipelined' },
  { num: '15.6', dir: 'lab6_linear', name: 'linear layer (M3)' },
  { num: '15.7', dir: 'lab7_bughunt', name: 'bug hunt' },
]);
const [l1, l2, l3, l4, l5, l6, l7] = LABS;
const STAGES = [l1, l2, l3, l4, l5];

// Identifier colours. Labs 15.3 to 15.5 call their warp-width template parameter W, and
// labs 15.6 and 15.7 call the weight W, so each family has its own map.
const TIERS_GEMM: Record<string, Tier> = {
  A: 'global', B: 'global', C: 'global', A_s: 'shared', B_s: 'shared', C_r: 'reg', a_r: 'reg', b_r: 'reg',
};
const TIERS_LINEAR: Record<string, Tier> = {
  X: 'global', W: 'global', Y: 'global', A_s: 'shared', B_s: 'shared', C_r: 'reg', a_r: 'reg', b_r: 'reg',
};

// ---------------------------------------------------------------- reading the benchmark rows
// Labs 15.1 to 15.5 (labs/ch15/shared/stage_driver.hpp) record, per shape, the rows
// tiled_5_3 (lab 5.3's reference kernel), previous (the stage before, from 15.2 on), stage
// and vendor, each with m, k, n, seconds and gflops; plus one pipeline row (Metal) or
// occupancy row (CUDA, HIP). Lab 15.6 records m2, m3 and vendor with batch, in, out.
const CUBE = { m: 2048, k: 2048, n: 2048 };
const SHORT_K = { m: 4096, k: 128, n: 4096 };
const LAYER_1 = { m: 1, k: 1024, n: 4096 };
const shapeText = (w: { m: number; k: number; n: number }) => `${w.m} × ${w.k} × ${w.n}`;
const gflops = (r?: BenchRow) => num(r?.gflops);

/** One stage row over another of the same run and shape, as GFLOP/s over GFLOP/s. */
const stageRatio =
  (lab: Lab, top: string, bottom: string, where: { m: number; k: number; n: number }, topName: string, bottomName: string) =>
  (reports: Reports) =>
    measure(reports, lab, (s: BenchSet) => {
      const a = gflops(benchRow(s, top, where));
      const b = gflops(benchRow(s, bottom, where));
      if (!a || !b) return null;
      return { value: a / b, detail: `${topName} ${fmt(a)} against ${bottomName} ${fmt(b)} GFLOP/s at ${shapeText(where)}` };
    });

/** Lab 15.6: M3 over M2 at batch 1 on the 1024 → 4096 layer. */
const m3OverM2Batch1 = (reports: Reports) =>
  measure(reports, l6, (s: BenchSet) => {
    const where = { batch: 1, in: 1024, out: 4096 };
    const a = gflops(benchRow(s, 'm3', where));
    const b = gflops(benchRow(s, 'm2', where));
    if (!a || !b) return null;
    return { value: a / b, detail: `${fmt(a)} against ${fmt(b)} GFLOP/s, batch 1, 1024 → 4096 layer` };
  });

// ---------------------------------------------------------------- results columns
const SHAPE_NAME: Record<string, string> = { square: 'square', layer: '1024 → 4096 layer', short_k: 'short K' };
const stageKernel: Col = {
  head: 'Kernel measured',
  cell: (r) =>
    r.name === 'tiled_5_3'
      ? 'lab 5.3 (T = 16)'
      : r.name === 'previous'
        ? `lab ${String(r.kernel)} (the stage before)`
        : r.name === 'stage'
          ? `this stage (${String(r.kernel)})`
          : r.name === 'vendor'
            ? 'vendor library'
            : r.name,
};
const shapeCol: Col = { head: 'Shape', cell: (r) => SHAPE_NAME[String(r.shape)] ?? String(r.shape ?? '') };
const mknCol: Col = { head: 'M × K × N', num: true, cell: (r) => `${String(r.m)} × ${String(r.k)} × ${String(r.n)}` };
const STAGE_ROWS = new Set(['tiled_5_3', 'previous', 'stage', 'vendor']);
const stageFilter = (r: BenchRow) => STAGE_ROWS.has(r.name);
const stageCols = (width: boolean): Col[] => [
  stageKernel,
  shapeCol,
  mknCol,
  ...(width ? [rawCol('Width', 'width')] : []),
  msCol,
  numCol('GFLOP/s', 'gflops'),
];
const linearKernel: Col = {
  head: 'Kernel measured',
  cell: (r) =>
    r.name === 'm2'
      ? 'M2 (lab 6.4, T = 16)'
      : r.name === 'm3'
        ? 'M3 (this lab)'
        : r.name === 'vendor'
          ? String(r.library ?? 'vendor library')
          : r.name,
};

// ---------------------------------------------------------------- what the device said about each kernel
/** Lab 15.6 spells the Metal pipeline's fields as Metal does; labs 15.1 to 15.5 in snake case. */
const pipelineOf = (r: BenchRow) => ({
  cap: num(r.max_threads_per_threadgroup) ?? num(r.maxTotalThreadsPerThreadgroup),
  simd: num(r.simd_width) ?? num(r.threadExecutionWidth),
  bytes: num(r.static_threadgroup_bytes) ?? num(r.staticThreadgroupMemoryLength),
});

/** The pipeline row (Metal) or occupancy row (CUDA, HIP) of a lab's benchmark, per backend. */
function KernelFacts({ lab }: { lab: Lab }) {
  const { reports, loading } = useLabs();
  const sets = preferredBench(reports, lab);
  const lines = sets.flatMap((s) =>
    s.rows
      .filter((r) => r.name === 'pipeline' || r.name === 'occupancy')
      .map((r) => {
        if (r.name === 'pipeline') {
          const p = pipelineOf(r);
          return {
            key: s.backend,
            text: `${s.backend} (${s.device}): the compiled pipeline accepts at most ${p.cap ?? '?'} threads per threadgroup, runs SIMD-groups of ${p.simd ?? '?'}, and uses ${p.bytes != null ? fmt(p.bytes) : '?'} bytes of static threadgroup memory.`,
          };
        }
        const blocks = num(r.blocks_per_sm);
        const regs = num(r.registers_per_thread);
        return {
          key: s.backend,
          text:
            blocks == null || blocks < 0
              ? `${s.backend} (${s.device}): the occupancy query was not available in this build.`
              : `${s.backend} (${s.device}): the runtime fits ${blocks} block${blocks === 1 ? '' : 's'} of ${String(r.threads_per_block ?? 256)} threads per SM${regs != null ? `, at ${regs} registers per thread` : ''}.`,
        };
      }),
  );
  return (
    <p className="status-line" data-kernel-facts={lab.num}>
      {loading
        ? 'Reading reports.'
        : lines.length
          ? lines.map((l) => <span key={l.key}>{l.text} </span>)
          : 'The benchmark also records what the device says about the compiled kernel (blocks per SM on CUDA and HIP, the pipeline’s limits on Metal); there is no such row on this machine yet.'}
    </p>
  );
}

// ---------------------------------------------------------------- the ladder
/** Every rung at 2048 cubed: each stage in its own run, and against the rung below in the same run. */
function Ladder() {
  const { reports, loading } = useLabs();
  const backends = [...new Set(STAGES.flatMap((l) => preferredBench(reports, l).map((s) => s.backend)))];
  const rungs = backends.map((b) => {
    const setOf = (l: Lab) => preferredBench(reports, l).find((s) => s.backend === b);
    const rows = STAGES.map((l, i) => {
      const s = setOf(l);
      const mine = gflops(benchRow(s, 'stage', CUBE));
      const below = gflops(benchRow(s, i === 0 ? 'tiled_5_3' : 'previous', CUBE));
      return { lab: l, s, mine, ratio: mine && below ? mine / below : undefined };
    });
    const first = setOf(l1);
    const last = STAGES.map(setOf).filter(Boolean).pop();
    return {
      b,
      device: rows.find((r) => r.s)?.s?.device ?? b,
      base: gflops(benchRow(first, 'tiled_5_3', CUBE)),
      vendor: gflops(benchRow(last, 'vendor', CUBE)),
      rows,
    };
  });
  const any = rungs.some((r) => r.rows.some((x) => x.mine));
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
      {loading ? (
        <p className="muted">Reading reports.</p>
      ) : !any ? (
        <p className="muted" data-ladder="none">
          No stage has a benchmark at {shapeText(CUBE)} on this machine yet. Run{' '}
          <code>python3 labs/pmpp.py run 15.1 --solution --bench</code> (and 15.2 to 15.5) on a machine with a GPU,
          then reload.
        </p>
      ) : (
        rungs.map((g) => (
          <table className="data" key={g.b} data-ladder={g.b}>
            <thead>
              <tr>
                <th>
                  {g.b} <span className="muted">({g.device})</span>, {shapeText(CUBE)}
                </th>
                <th className="num">GFLOP/s</th>
                <th className="num">× the rung below, same run</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>lab 5.3 (T = 16), from lab 15.1&rsquo;s run</td>
                <td className="num">{g.base ? fmt(g.base) : ''}</td>
                <td className="num" />
              </tr>
              {g.rows.map(({ lab, s, mine, ratio }) => (
                <tr key={lab.num} data-rung={lab.num}>
                  <td>
                    lab {lab.num}, {lab.name}
                    {s ? <span className="muted"> ({s.variant === 'work' ? 'yours' : 'reference solution'}, {s.date})</span> : null}
                  </td>
                  <td className="num">{mine ? fmt(mine) : ''}</td>
                  <td className="num">{ratio ? fmt(ratio) : ''}</td>
                </tr>
              ))}
              <tr>
                <td>vendor library, from lab 15.5&rsquo;s run</td>
                <td className="num">{g.vendor ? fmt(g.vendor) : ''}</td>
                <td className="num" />
              </tr>
            </tbody>
          </table>
        ))
      )}
    </div>
  );
}

// ---------------------------------------------------------------- the race backend
type RaceResult = RunResult & { tsan?: unknown[] };
/** Every lab of the page on the race backend, which the status table does not show. */
function RaceTable() {
  const { reports, loading } = useLabs();
  const run = reports.race;
  const line = (l: Lab, variant: 'solution' | 'work') => {
    const rs = (run?.results.filter((x) => x.lab === l.id && x.variant === variant) ?? []) as RaceResult[];
    const tests = rs.flatMap((x) => x.tests);
    const races = rs.reduce((n, x) => n + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0);
    const s = labStatus(run, l.id, variant);
    const text =
      s === 'none'
        ? variant === 'work'
          ? 'not started'
          : 'no run on this machine'
        : `${s === 'pass' ? 'passes' : 'not passing'}: ${tests.filter((t) => t.pass).length} of ${tests.length} tests, ${races ? `${races} ThreadSanitizer report${races === 1 ? '' : 's'}` : 'no race reported'}`;
    return { s, text };
  };
  if (loading) return <p className="muted">Reading reports.</p>;
  if (!run) return <p data-race="none">No run on the race backend on this machine yet.</p>;
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
      <table className="data" data-table="race">
        <thead>
          <tr>
            <th>On the race backend</th>
            <th>Reference solution</th>
            <th>Your work</th>
          </tr>
        </thead>
        <tbody>
          {LABS.map((l) => {
            const sol = line(l, 'solution');
            const mine = line(l, 'work');
            return (
              <tr key={l.id} data-race-lab={l.num} data-race-solution={sol.s} data-race-work={mine.s}>
                <td>
                  Lab {l.num}, {l.name}
                </td>
                <td>{sol.s === 'pass' ? `emulator-checked; ${sol.text}` : sol.text}</td>
                <td>{mine.text}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------- milestone M3
function MilestoneStatus() {
  const { reports, loading } = useLabs();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const pass = backends.filter((b) => labStatus(reports[b], l6.id, 'work') === 'pass');
  const name = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  return (
    <p data-milestone={loading ? 'loading' : pass.join(',') || 'none'}>
      {loading ? (
        'Reading reports.'
      ) : !backends.length ? (
        'No lab results are recorded on this machine yet.'
      ) : pass.length ? (
        <>
          Your own <code>launchLinear</code> passes lab 15.6&rsquo;s tests on {pass.map(name).join(' and ')} on this
          machine: milestone M3 is in place there. How it compares with M2 and the library is the benchmark&rsquo;s to
          say.
        </>
      ) : (
        <>
          Your own <code>launchLinear</code> does not pass lab 15.6&rsquo;s tests on any backend of this machine yet.
          Until it does, the project&rsquo;s linear layer is M2.
        </>
      )}
    </p>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS}>
      <LabsIntro>
        Seven labs, each in its own directory under <code>labs/ch15/</code>. Labs 15.1 to 15.5 are one ladder in the
        book&rsquo;s layout, C = A B with both inputs row-major: each starts from the previous lab&rsquo;s solution with
        one part taken out, and each benchmark times your stage beside lab 5.3&rsquo;s kernel, the stage before it and the
        vendor&rsquo;s GEMM. Lab 15.6 is milestone M3, the same kernel as the project&rsquo;s linear layer. Lab 15.7
        hides three defects in it for you to find. Code you do not edit, the shared tests and benchmark and the vendor
        calls, sits in <code>labs/ch15/shared/</code> and in lab 15.6&rsquo;s <code>provided/</code> folder.
      </LabsIntro>
      <RunningLabs
        example="15.1"
        alsoShownIn={
          <>
            the <a href={href('ch15', 'start')}>start page</a> and sections <a href={href('ch15', '15.7')}>15.7</a> and{' '}
            <a href={href('ch15', '15.9')}>15.9</a>
          </>
        }
      />
      <BackendsAndEmulator />
      <p>
        Every lab here is also built for the emulator&rsquo;s second flavour, <code>race</code>, which runs one block&rsquo;s
        threads at the same time under ThreadSanitizer and names the two source lines of any unsynchronised pair of
        accesses. Ask for it with <code>--backend race</code>. Its shapes are small, one to four blocks, because a
        256-thread block under ThreadSanitizer is slow. The status tables above show the first flavour; this one shows
        the second.
      </p>
      <RaceTable />
      <p>
        What the emulators cannot see matters more in this chapter than in any before it. Labs 15.2 to 15.5 change
        nothing a test can observe: register tiling, the store pattern, padding and pipelining all give the same C.
        They differ only in speed, so on the emulators they are correctness checks and nothing else, and the prediction
        for each is settled by a GPU. One correctness gap too: a CPU does not mind a misaligned <code>float4</code>, so a
        vector access without its alignment condition passes on <code>emu</code> and fails on a GPU.
      </p>
      <p>
        The benchmarks time the vendor&rsquo;s GEMM beside yours: <code>cublasSgemm</code> on CUDA,{' '}
        <code>hipblasSgemm</code> on HIP, <code>MPSMatrixMultiplication</code> on Metal, each checked against the CPU
        reference first. All three compute in FP32 unless told otherwise, and the benchmark prints the math mode. The
        cuBLAS and hipBLAS calls have never been built on a CUDA or ROCm machine (unverified). If they do not build,
        configure once with <code>cmake -S labs -B labs/build</code> and <code>-DPMPP_CH15_VENDOR=OFF</code> for labs
        15.1 to 15.5, <code>-DPMPP_CH15_VENDOR_BLAS=OFF</code> for lab 15.6, or{' '}
        <code>-DPMPP_CH15_OCCUPANCY_API=OFF</code> if the occupancy query is what fails. CMake keeps the setting for
        every later <code>pmpp.py</code> run.
      </p>
      <Today checked="2026-11-20 (notes/today/ch15-facts.md A10, B6, C6)">
        <p>
          cuBLAS and hipBLAS are column-major, so a row-major product is asked for as Cᵀ = Bᵀ Aᵀ. TF32 and BF16x9 (cuBLAS),
          XF32 (hipBLAS) and reduced precision (MPS) are all opt-in.
        </p>
      </Today>
      <VerificationStatus />

      {/* ------------------------------------------------------------ 15.1 */}
      <h2>Lab 15.1: the 128 × 128 coarsened kernel</h2>
      <p>
        You write the book&rsquo;s large-tile kernel for C[M × N] = A[M × K] B[K × N], row-major, any sizes. A block of
        256 threads computes a 128 × 128 tile of C. Each thread owns an 8 × 8 patch of it and keeps the patch in
        registers, while the inner dimension streams through shared memory 8 columns of A and 8 rows of B at a time. Four
        helpers, <code>clear</code>, <code>loadTile</code>, <code>mm</code> and <code>writeTile</code>, then the kernel
        and its launch. Section <a href={href('ch15', '15.3')}>15.3</a> walks through each of them.
      </p>
      <Source section="15.3" pages="352–357" figs="Figs. 15.3 to 15.7" />
      <p>
        Where the lab differs from the page, on purpose: the block is one-dimensional, 256 threads (the prose says 16 ×
        16, but Fig. 15.3 reads only <code>threadIdx.x</code>); <code>A_s</code> and <code>B_s</code> are declared once,
        at the top of the kernel; sizes are <code>int</code> and the arguments are <code>(A, B, C, M, K, N)</code>, lab
        5.3&rsquo;s order, so milestone M2&rsquo;s call sites work unchanged; and the helpers are <code>static</code>,
        because later labs link this solution into their benchmark and the emulator&rsquo;s linker would silently merge
        two non-static inline functions with one name.
      </p>
      <Beyond>
        <p>
          <code>__launch_bounds__(256)</code> is ours: without it HIP compiles for the largest block a device accepts
          and may squeeze registers (ch15-facts B3). On Metal the helpers are <code>METAL_FUNC</code>, Apple&rsquo;s
          header macro, not in the specification (C4).
        </p>
      </Beyond>
      <LabFiles
        lab={l1}
        files={[
          ['kernel.cu', <><code>clear</code>, <code>loadTile</code>, <code>mm</code>, <code>writeTile</code>, <code>gemmCoarseKernel</code> and <code>launchGemmCoarse</code></>],
          ['kernel.metal', <><code>gemm_coarse</code> and the same four helpers</>],
          ['launch.mm', <><code>encodeGemmCoarse</code>: the threadgroups and the bindings</>],
        ]}
      />
      <p>
        Run <code>python3 labs/pmpp.py run 15.1</code>, starting with <code>--backend emu</code>. The shapes each break
        one thing: a single element, one block with one tile pair, three tile pairs, partial tiles in every direction
        with K and N not multiples of 4, one decode row across three column blocks, K = 1, four full blocks, a narrow N,
        and K = 0, an empty sum whose C must come out all zeros. A GPU adds a 1024 square and three large awkward
        shapes. C starts as NaN and is followed by a guard band of 128
        rows holding a sentinel, so an element you never write and a store past the last row both fail, and the message
        says which. A and B hold exactly their floats, so on <code>emu</code> AddressSanitizer names the line of any
        read past either. On Metal the driver first checks that the pipeline accepts 256 threads per threadgroup, since
        Metal may set a lower limit than the kernel asks for.
      </p>
      <p>
        With <code>--bench</code>, lab 5.3&rsquo;s reference solution, your kernel and the vendor&rsquo;s GEMM take turns
        on squares of 1024, 2048 and 4096, the 1024 → 4096 layer at batch 1, 16, 256 and 1024, and the short-K shape
        4096 × 128 × 4096. Two predictions first.
      </p>
      <LabPrediction
        id="ch15/labs/p-lab1-speedup"
        prompt={
          <>
            At {shapeText(CUBE)}, what speed-up of your kernel over lab 5.3&rsquo;s will you measure? Weigh three things.
            The model goes from 4 FLOP/B for lab 5.3&rsquo;s 16 × 16 tiles to 32 for 128 × 128: eight times less global
            traffic requested. The inner loop of Fig. 15.6 still reads two values from shared memory for every
            multiply-add, as lab 5.3&rsquo;s does. And caches already served much of lab 5.3&rsquo;s traffic on the
            machine this course was piloted on.
          </>
        }
        unit="× lab 5.3"
        measure={stageRatio(l1, 'stage', 'tiled_5_3', CUBE, 'lab 15.1', 'lab 5.3')}
        reading={
          <>
            The model&rsquo;s factor of 8 counts bytes requested from global memory, not time. A ratio well under it says
            something other than DRAM traffic was limiting lab 5.3 here, or now limits this kernel: section{' '}
            <a href={href('ch15', '15.4')}>15.4</a> names the next suspect.
          </>
        }
      />
      <LabPrediction
        id="ch15/labs/p-lab1-batch1"
        prompt={
          <>
            At batch 1 on the 1024 → 4096 layer ({shapeText(LAYER_1)}), your kernel&rsquo;s speed as a multiple of lab
            5.3&rsquo;s. Count the blocks each one launches (the benchmark prints them) and compare with your
            device&rsquo;s number of SMs from <code>python3 labs/pmpp.py doctor</code>. Below 1 means slower.
          </>
        }
        unit="× lab 5.3"
        measure={stageRatio(l1, 'stage', 'tiled_5_3', LAYER_1, 'lab 15.1', 'lab 5.3')}
        reading={
          <>
            Lab 5.3 launches 256 blocks of 16 × 16 for this shape, yours 32 blocks of 128 × 128, each with 127 rows of
            nothing to do. Metal reports no SM count, so on a Mac the block count is the only half of the comparison you
            can read. Section <a href={href('ch15', '15.2')}>15.2</a> has the other reason: at m = 1 no tile shape raises
            the intensity.
          </>
        }
      />
      <LabResults lab={l1} filter={stageFilter} cols={stageCols(false)} />
      <KernelFacts lab={l1} />
      <ReferenceSolution lab={l1} tiers={TIERS_GEMM} />

      {/* ------------------------------------------------------------ 15.2 */}
      <h2>Lab 15.2: register tiling, the loops interchanged</h2>
      <p>
        Your work starts as lab 15.1&rsquo;s solution, renamed, with the body of <code>mm</code> removed. Rewrite it the
        way Fig. 15.9 does: the inner-dimension loop <code>i</code> becomes the outermost one; for each <code>i</code>,
        copy the thread&rsquo;s 8 values of column <code>i</code> of its rows of <code>A_s</code> into <code>a_r</code>,
        and its 8 values of row <code>i</code> of its columns of <code>B_s</code> into <code>b_r</code>; then add the
        outer product to all 64 elements of the tile. Unroll all five loops completely, or <code>a_r</code>,{' '}
        <code>b_r</code> and the tile leave the registers.
      </p>
      <Source section="15.4" pages="357–360" figs="Figs. 15.8 and 15.9" />
      <LabFiles
        lab={l2}
        files={[
          ['kernel.cu', <><code>mm</code> only</>],
          ['kernel.metal', <><code>mm</code> only</>],
        ]}
      />
      <p>
        <code>work/launch.mm</code> is complete: register tiling changes the kernel, not the launch. The tests are lab
        15.1&rsquo;s; the untouched starter builds and fails, because its <code>mm</code> adds nothing and C comes out
        all zeros. The benchmark adds lab 15.1&rsquo;s solution as the stage before, copied into the build when you
        configure.
      </p>
      <LabPrediction
        id="ch15/labs/p-lab2-speedup"
        prompt={
          <>
            At {shapeText(CUBE)}, your speed-up over lab 15.1. Count first: per thread and tile pair, Fig. 15.6 reads
            shared memory 1024 times and Fig. 15.9 128 times. Then weigh that the strip reads still conflict in this
            thread layout, on 32 banks the A strip 2-way and the B strip 4-way, and that the stores have not changed.
          </>
        }
        unit="× lab 15.1"
        measure={stageRatio(l2, 'stage', 'previous', CUBE, 'lab 15.2', 'lab 15.1')}
        reading={
          <>
            A compiler that fully unrolls Fig. 15.6 sees every index as a constant and may already keep repeated reads in
            registers on its own (our reasoning, not a vendor statement). The interchange makes the reuse explicit;
            whether it was missing is what this number tells you. On Metal, Apple documents neither banks nor registers
            per thread, so nothing documented predicted it.
          </>
        }
      />
      <LabResults lab={l2} filter={stageFilter} cols={stageCols(false)} />
      <KernelFacts lab={l2} />
      <ReferenceSolution lab={l2} tiers={TIERS_GEMM} />

      {/* ------------------------------------------------------------ 15.3 */}
      <h2>Lab 15.3: coalesced stores, four 4 × 4 sub-tiles and float4 stores</h2>
      <p>
        In labs 15.1 and 15.2 each thread stores an 8 × 8 patch, so in one store instruction neighbouring threads of a
        warp write addresses 8 floats apart. Your work starts as lab 15.2&rsquo;s solution, made a template over the
        warp width <code>W</code>. Re-map the threads as section <a href={href('ch15', '15.5')}>15.5</a> describes, and
        store each row of a sub-tile with one <code>float4</code>. For W = 32 that is the book&rsquo;s 8 warps as 2 × 4,
        warp tiles of 64 × 32, quadrants of 32 × 16 and lanes as 8 × 4; for W = 64 the same rules give 4 warps as 2 × 2,
        warp tiles of 64 × 64 and quadrants of 32 × 32. The book gives no code for this section: it is Exercises 3 and
        2.
      </p>
      <Source section="15.5" pages="360–361" figs="Fig. 15.10; Exercises 2 and 3, p. 370" />
      <p>
        Never write the warp width as a number inside the kernel. <code>launchGemmRemap</code> reads it from the device
        once and calls the template instance for it; <code>launchGemmRemapWidth</code>, which the tests call with both
        widths, is written for you. On Metal the host compiles the kernel with <code>SIMD_WIDTH</code> defined and binds
        C a second time as <code>device float4*</code>, as lab 6.2 does.
      </p>
      <Today checked="2026-11-20 (ch15-facts B1, C1; ch06-facts C3)">
        <p>
          AMD: warpSize &ldquo;can not be assumed to be a compile-time constant on the host&rdquo;; its recommended route
          is a host-chosen template instance. Apple: the width comes only from the pipeline&rsquo;s{' '}
          <code>threadExecutionWidth</code>.
        </p>
      </Today>
      <LabFiles
        lab={l3}
        files={[
          ['kernel.cu', <>the mapping, <code>mm</code> over the four sub-tiles, <code>writeTile</code> with <code>float4</code> stores, and <code>launchGemmRemap</code></>],
          ['kernel.metal', <><code>gemm_remap</code>, with the width as <code>SIMD_WIDTH</code></>],
          ['launch.mm', <><code>encodeGemmRemap</code>, binding C twice</>],
        ]}
      />
      <p>
        The tests are lab 15.1&rsquo;s, each run with the layout for W = 32 and for W = 64 on every backend, the
        emulators included: the layout is index arithmetic, so whether it is right does not depend on the
        hardware&rsquo;s width. On CUDA and HIP the launch that reads the device&rsquo;s width runs too. Shapes with N =
        1, 3, 7, 9, 131 and 511 need the scalar fallback. The untouched starter builds and fails: its{' '}
        <code>writeTile</code> stores nothing, so C stays NaN.
      </p>
      <LabPrediction
        id="ch15/labs/p-lab3-shortk"
        prompt={
          <>
            On the short-K shape {shapeText(SHORT_K)}, your speed-up over lab 15.2. Stores happen once per output however
            long K is: with K = 128 there are 16 tile pairs to load and compute for every tile stored, with K = 2048 there
            are 256.
          </>
        }
        unit="× lab 15.2"
        measure={stageRatio(l3, 'stage', 'previous', SHORT_K, 'lab 15.3', 'lab 15.2')}
        reading={
          <>
            The remap fixes the stores and removes the B strip&rsquo;s 4-way bank conflict, but raises the A strip&rsquo;s
            from 2-way to 8-way on 32 banks: eight lane rows read rows of <code>A_s</code> 4 apart (ch15-facts E1). Lab
            15.4 adds the padding that removes it. Apple documents no coalescing rule, so on a Mac this is a plain
            measurement.
          </>
        }
      />
      <LabPrediction
        id="ch15/labs/p-lab3-speedup"
        prompt={<>And at {shapeText(CUBE)}, where 256 tile pairs share each store: your speed-up over lab 15.2.</>}
        unit="× lab 15.2"
        measure={stageRatio(l3, 'stage', 'previous', CUBE, 'lab 15.3', 'lab 15.2')}
        reading={
          <>
            Compare the two numbers before you explain either. A store pattern can only matter in proportion to the time
            spent storing, and that share falls as K grows.
          </>
        }
      />
      <LabResults lab={l3} filter={stageFilter} cols={stageCols(true)} />
      <KernelFacts lab={l3} />
      <ReferenceSolution lab={l3} tiers={TIERS_GEMM} />

      {/* ------------------------------------------------------------ 15.4 */}
      <h2>Lab 15.4: padding and vector loads</h2>
      <p>
        Your work starts as lab 15.3&rsquo;s solution with <code>A_s</code> unpadded and the body of{' '}
        <code>loadTile</code> removed. Two changes. First, padding: give <code>A_s</code> the leading dimension bK + 1
        everywhere it is indexed, which is its declaration, the shared store in <code>loadTile</code>, the row stride of
        the strip reads in <code>mm</code>, and the thread&rsquo;s start offset in the call to <code>mm</code>. The book
        names the first three and not the last (section <a href={href('ch15', '15.6')}>15.6</a>). Second, Exercise 1:
        one <code>float4</code> load per thread per tile, when it is safe, and the scalar path with zero fill otherwise.
        Store into shared memory one float at a time: a padded row of <code>A_s</code> is 9 floats, so a{' '}
        <code>float4</code> store there would not be aligned.
      </p>
      <Source section="15.6" pages="361–363" figs="Fig. 15.11; Exercise 1, p. 370; vector loads at low occupancy, §15.7 p. 364" />
      <p>
        The book writes the declaration <code>A_s[bM*(bK + 1)]</code>. The lab gives the padded leading dimension a
        name, <code>#define LDA_S (bK + 1)</code>, writes <code>A_s[bM * LDA_S]</code>, and uses the same name in every
        index into <code>A_s</code>, so that no index can keep the old width.
      </p>
      <LabFiles
        lab={l4}
        files={[
          ['kernel.cu', <>the padded declaration, every index into <code>A_s</code>, and <code>loadTile</code></>],
          ['kernel.metal', <><code>gemm_padded</code>, where the vector loads go through <code>device const float4*</code> views of A and B</>],
          ['launch.mm', <><code>encodeGemmPadded</code>, with the extra <code>float4</code> bindings</>],
        ]}
      />
      <p>
        The tests are lab 15.3&rsquo;s, at both widths; shapes with K or N not a multiple of 4 run the scalar fallback.
        The untouched starter builds and fails: <code>loadTile</code> writes nothing, so <code>mm</code> multiplies
        whatever was in shared memory. The missing start offset is caught on its own: with the padded declaration and
        leading dimension but <code>&amp;A_s[tRow * bK]</code>, every shape with 5 rows or more fails. Shapes with 1, 3
        or 4 rows pass, because all their rows belong to threads whose sub-tiles start at row 0 of the tile, where the
        two offsets agree. A test that used only small matrices would have missed it.
      </p>
      <p>
        What the two changes should buy: on 32 banks the A-strip conflict of lab 15.3&rsquo;s layout goes from 8-way to
        none, because rows 4 apart now start 36 words apart, in banks 0, 4, ..., 28. And vector loads cut each
        thread&rsquo;s load instructions per tile pair from 8 to 2, the book&rsquo;s remedy for a kernel that runs too
        few threads to keep memory busy.
      </p>
      <Today checked="2026-11-20 (ch15-facts A2, B2, C2, E1)">
        <p>
          Any odd leading dimension removes this conflict on 32 banks; on 64 banks any not divisible by 4 does (our
          arithmetic). MLX pads its Metal tiles by 16 bytes, a practice with no Apple documentation behind it.
        </p>
      </Today>
      <LabPrediction
        id="ch15/labs/p-lab4-speedup"
        prompt={<>At {shapeText(CUBE)}, what speed-up over lab 15.3 will you measure?</>}
        unit="× lab 15.3"
        measure={stageRatio(l4, 'stage', 'previous', CUBE, 'lab 15.4', 'lab 15.3')}
        reading={
          <>
            Two changes in one stage: if the number surprises you, take one out and measure again. On AMD devices with 64
            banks the conflict figures are our model; on Metal, Apple documents no threadgroup-memory banks at all, so
            this is a measurement with no prediction from the vendor.
          </>
        }
      />
      <LabResults lab={l4} filter={stageFilter} cols={stageCols(true)} />
      <KernelFacts lab={l4} />
      <ReferenceSolution lab={l4} tiers={TIERS_GEMM} />

      {/* ------------------------------------------------------------ 15.5 */}
      <h2>Lab 15.5: double buffering and software pipelining</h2>
      <p>
        Your work starts as lab 15.4&rsquo;s solution with its tile loop taken out. Put it back so that iteration{' '}
        <code>tile</code> computes with pair <code>tile</code> while it loads pair <code>tile + 1</code> into a second
        set of buffers, with one barrier per iteration instead of two: two pairs of shared tiles,{' '}
        <code>A_s[2][...]</code> and <code>B_s[2][...]</code>; a prologue that loads pair 0 and waits; a loop over{' '}
        <code>tile &lt; numTiles - 1</code> that runs <code>mm</code> on pair <code>tile</code>, loads pair{' '}
        <code>tile + 1</code> into the other buffers, and waits; and an epilogue that runs <code>mm</code> on the last
        pair.
      </p>
      <Source section="15.8" pages="364–368" figs="Figs. 15.12 to 15.14" />
      <p>
        Fig. 15.14 is a skeleton: its arguments are elided, its buffers undeclared, and its pointer types do not match
        the one-dimensional tiles of Fig. 15.3. The book swaps a current and a next pointer each iteration; choosing a
        pair&rsquo;s buffers by its parity, <code>tile &amp; 1</code>, as lab 6.5 does, is the same alternation. Two
        cases the figure leaves to you: what <code>numTiles - 1</code> does when K is 0 (a test case has K = 0 and a
        30-second limit), and which buffers the epilogue reads. Section <a href={href('ch15', '15.8')}>15.8</a> steps through both.
      </p>
      <LabFiles
        lab={l5}
        files={[
          ['kernel.cu', <>the double buffers and the tile loop in <code>gemmPipelinedKernel</code></>],
          ['kernel.metal', <>the same in <code>gemm_pipelined</code></>],
        ]}
      />
      <p>
        <code>work/launch.mm</code> is complete. Before you write anything, answer from Fig. 15.12: which of lab
        15.4&rsquo;s two barriers guards a true dependence and which a false one, and why can a thread that runs ahead
        never overwrite a pair its neighbours are still reading? Then run <code>--backend emu</code>, then{' '}
        <code>--backend race</code>, then the rest. On a GPU every case is launched 10 times with C refilled with NaN, and
        a failure says in how many launches the result was wrong. A removed loop barrier was checked to fail on its own:
        every case with more than one tile pair fails on <code>emu</code> under all three thread orders, ThreadSanitizer
        names the store in <code>loadTile</code> and the read in <code>mm</code>, and on Metal, the only GPU it was
        checked on, some cases failed in only a few of the 10 launches. The untouched starter builds and fails: without the loop, C comes out 0.
      </p>
      <LabPrediction
        id="ch15/labs/p-lab5-speedup"
        prompt={
          <>
            At {shapeText(CUBE)}, your speed-up over lab 15.4. First read lab 15.4&rsquo;s line above about what the
            device said: if two or more blocks fit on an SM, other blocks were already overlapping their loads with your
            block&rsquo;s arithmetic. Weigh too that the overlap is the compiler&rsquo;s to make, and that shared memory per
            block doubles, to 17,408 bytes with the padding.
          </>
        }
        unit="× lab 15.4"
        measure={stageRatio(l5, 'stage', 'previous', CUBE, 'lab 15.5', 'lab 15.4')}
        reading={
          <>
            The source only puts the loads of pair <code>tile + 1</code> and the arithmetic of pair <code>tile</code> in
            one iteration with no barrier between them; nothing makes the compiler interleave them. Then try the other
            order inside the loop, loads first, then <code>mm</code>. It is just as correct (the tests will tell you) and
            it hands the compiler different code. Measure both. Whatever you find is a fact about your device and your
            compiler, not a rule.
          </>
        }
      />
      <p>
        Libraries take the overlap out of the compiler&rsquo;s hands with copies the hardware runs in the background.
        Section <a href={href('ch15', '15.9')}>15.9</a> says which backends have them.
      </p>
      <Today checked="2026-11-20 (ch15-facts A7, A8, A12, B5; ch06-facts C7)">
        <p>
          CUDA can copy from global to shared memory without registers (LDGSTS, compute capability 8.0 and later) and
          Hopper adds TMA; HIP and Metal document no usable asynchronous copy for this. The synchronous loop of this lab
          is the form every backend has.
        </p>
      </Today>
      <LabResults lab={l5} filter={stageFilter} cols={stageCols(true)} />
      <KernelFacts lab={l5} />
      <ReferenceSolution lab={l5} tiers={TIERS_GEMM} />

      <h3>The ladder so far</h3>
      <p>
        Each rung at {shapeText(CUBE)}, as its own lab&rsquo;s benchmark measured it, with the ratio to the rung below
        taken from the same run. The runs of different labs were made at different times, and on a busy machine two
        runs of one kernel can differ by more than neighbouring rungs do: read the ratio column, measured side by side,
        before the GFLOP/s column.
      </p>
      <Ladder />

      {/* ------------------------------------------------------------ 15.6 */}
      <h2>Lab 15.6: the fast linear layer (milestone M3)</h2>
      <p>
        You write <code>launchLinear(X, W, Y, batch, in, out)</code>, computing Y = X Wᵀ with W stored (out, in)
        row-major, as <code>nn.Linear</code> and ggml store a weight. It is lab 6.4&rsquo;s signature exactly, so this
        kernel drops in where M2 was (on Metal, <code>encodeLinear</code> loses M2&rsquo;s tile-width argument). Inside goes your lab 15.5 kernel adapted to the project&rsquo;s layout: in the
        book&rsquo;s names A = X (M = batch, K = in), B = Wᵀ (N = out), C = Y. <code>clear</code>, <code>loadTile</code>,{' '}
        <code>mm</code> and <code>writeTile</code> carry over. Three things are new: <code>loadTileT</code>, which reads
        the W tile the way W is stored and stores it transposed into <code>B_s[k][n]</code>; the pitch of{' '}
        <code>B_s</code>, which the starter sets to <code>bN</code>, correct and conflicting; and the names, since W is
        now the weight, the width parameter is <code>WARP</code>.
      </p>
      <Source section="15.3 to 15.8" pages="352–368" figs="Exercises 1 to 3, p. 370" />
      <p>
        Everything that adapts lab 15.5 to the project layout is ours. Storing W transposed keeps the B strip read a
        contiguous row, as in the book; it is lab 6.4&rsquo;s corner turning at the 128 × 8 scale. A pitch of bN + 4 =
        132 makes that transposed store conflict-free on 32 banks for both load mappings, which section{' '}
        <a href={href('ch15', '15.6')}>15.6</a>&rsquo;s explorer draws in its third mode, keeps rows 16-byte aligned, and
        is 16 bytes of padding, the amount MLX gives its tiles.
      </p>
      <Beyond>
        <p>
          The project layout is lab 6.4&rsquo;s, from PyTorch and ggml. The pitch analysis is our arithmetic with
          NVIDIA&rsquo;s bank rule (ch15-facts A2); Apple documents no banks (C2), so on Metal it is a measurement.
        </p>
      </Beyond>
      <LabFiles
        lab={l6}
        files={[
          ['kernel.cu', <><code>fastLinearKernel&lt;WARP&gt;</code> and its helpers, <code>loadTileT</code> and the pitch <code>LDB_S</code> included</>],
          ['kernel.metal', <><code>linear_m3</code></>],
          ['launch.mm', <><code>encodeLinear</code> is complete; it binds Y, X and W a second time, as <code>float4</code></>],
        ]}
      />
      <p>
        Handle <code>in == 0</code>: Fig. 15.14&rsquo;s <code>numTiles - 1</code> must not wrap. A case with in = 0
        checks it: Y must come out all zeros, within 30 seconds. The shapes (batch, in, out) run from one element to
        (1024, 1024, 1024) and a few tokens through a 4096-wide layer, each at warp widths
        32 and 64 on every backend, plus one case through <code>launchLinear</code> as the project calls it. Y starts as
        NaN and is followed by a guard band; the reference is y = x Wᵀ on the CPU in double, with a tolerance scaled by
        the sum of |x w| per output. On a GPU the provided vendor call is checked against the same reference: a failure
        there is the course&rsquo;s bug, not yours. The untouched starter builds and fails everything except those
        vendor checks.
      </p>
      <p>
        With <code>--bench</code>, three kernels take turns: M2 (lab 6.4&rsquo;s reference solution, T = 16, copied into
        the build when you configure), your M3, and the vendor library, on the 1024 → 4096 layer at batch 1, 4, 16, 64,
        256 and 1024 and on a 4096 → 4096 layer at batch 1, 16 and 256. The second layer is our choice, a 7B-class
        hidden size. The prediction at batch 256, your M3 as a percentage of the library, is in section{' '}
        <a href={href('ch15', '15.9')}>15.9</a> under &ldquo;Your kernel against the library&rdquo;; it reads the same
        rows. The one here is about batch 1.
      </p>
      <LabPrediction
        id="ch15/labs/p-lab6-batch1"
        prompt={
          <>
            At batch 1 on the 1024 → 4096 layer, your M3&rsquo;s speed as a multiple of M2&rsquo;s. All three kernels must
            read the whole weight matrix once; a 128-row block tile computes 128 rows of which 127 are padding. Below 1
            means M3 is slower.
          </>
        }
        unit="× M2"
        measure={m3OverM2Batch1}
        reading={
          <>
            Then rank all three from the table, library included. A library may pick a different kernel for each shape:
            llama.cpp switches to matrix-vector kernels up to batch 8, and PyTorch&rsquo;s own comment calls its gemv
            &ldquo;bw bound&rdquo; (ch15-facts D1, D4). Nothing guarantees M3 beats M2 at any batch on your device
            either: the book&rsquo;s argument is made for NVIDIA&rsquo;s registers and banks, and Apple documents neither
            (C2, C3).
          </>
        }
      />
      <LabResults
        lab={l6}
        filter={(r) => r.name === 'm2' || r.name === 'm3' || r.name === 'vendor'}
        cols={[linearKernel, rawCol('Batch', 'batch'), rawCol('In', 'in'), rawCol('Out', 'out'), msCol, numCol('GFLOP/s', 'gflops')]}
      />
      <KernelFacts lab={l6} />
      <ReferenceSolution lab={l6} tiers={TIERS_LINEAR} />
      <h3>The project: milestone M3</h3>
      <p>
        M3 keeps the conventions of M2: row-major <code>float</code> tensors with the batch first, weights stored (out,
        in), raw device pointers and explicit sizes, and <code>launchLinear(X, W, Y, batch, in, out)</code> beside the
        kernel. Every later milestone that needs a linear layer can call either one.
      </p>
      <MilestoneStatus />

      {/* ------------------------------------------------------------ 15.7 */}
      <h2>Lab 15.7: bug hunt, three defects in a fast linear layer</h2>
      <p>
        <code>work/kernel.cu</code> and <code>work/kernel.metal</code> hold a fast linear layer in the project&rsquo;s
        layout, lab 15.6&rsquo;s design, with three independent defects, the same in both files. It compiles, and some
        shapes pass. Find and fix all three. These are mistakes this chapter&rsquo;s kernel invites, and finding them in
        code you did not write is the skill you will need for your own.
      </p>
      <Source section="15.6 and 15.8" pages="361–368" figs="Fig. 15.12; Exercise 2, p. 370" />
      <LabFiles
        lab={l7}
        files={[
          ['kernel.cu', <><code>fastLinearKernel</code> and its helpers, with the three defects</>],
          ['kernel.metal', <><code>linear_m3</code>, with the same three</>],
          ['launch.mm', <>provided and correct; nothing to change</>],
        ]}
      />
      <p>The method matters more than the fixes:</p>
      <ol>
        <li>
          Before you run anything, read the kernel and write down, for each defect you suspect, which backends you
          expect to show it: <code>emu</code>, <code>race</code>, Metal, CUDA or HIP.
        </li>
        <li>
          <code>python3 labs/pmpp.py run 15.7 --backend emu</code>. Its thread orders are fixed, so a synchronisation
          defect fails every time, and AddressSanitizer stops at the first access outside an allocation and names the
          line.
        </li>
        <li>
          Fix one thing and run again; the failures change as you go. Read what each failure says: which rows are wrong,
          which shapes fail and which pass, whether the guard band after Y changed.
        </li>
        <li>
          <code>python3 labs/pmpp.py run 15.7 --backend race</code>. ThreadSanitizer names the two source lines whose
          accesses collide, which says where a barrier is missing. It sees only collisions that happened in that run,
          so a clean report is not a proof.
        </li>
        <li>
          Then on your GPU, where each case is launched ten times with Y back to NaN. Compare with what you wrote down
          in step 1: which defects did your GPU hide?
        </li>
      </ol>
      <p>
        The checks are lab 15.6&rsquo;s: the same shapes at warp widths 32 and 64, X and W allocated with exactly their
        floats, and the guard band after Y that no kernel may write. Each defect was checked to fail on its own on the
        emulators and on Metal.
      </p>
      <LabResults
        lab={l7}
        noBench="This lab has no benchmark. The result is the line above: every test passing on the emulators and on a device. As shipped, the kernel is meant to fail, and on the emulator it may abort."
      />
      <ReferenceSolution lab={l7} tiers={TIERS_LINEAR} />
    </LabsPage>
  );
}
