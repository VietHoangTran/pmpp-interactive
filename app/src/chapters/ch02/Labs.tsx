import type { Question } from '../../components/Ask';
import { Predict } from '../../components/Ask';
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
  numCol,
  rawCol,
  useLabs,
  type BenchSet,
  type Col,
} from '../../components/LabKit';
import { fmt, num, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 2's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch02', [
  { num: '2.1', dir: 'lab1_vecadd', name: 'vector addition' },
  { num: '2.2', dir: 'lab2_timing', name: 'where the time goes' },
  { num: '2.3', dir: 'lab3_elementwise', name: 'elementwise ops (M0)' },
  { num: '2.4', dir: 'lab4_bughunt', name: 'bug hunt' },
]);
const [l1, l2, l3, l4] = LABS;

// Identifier colours for the reference solutions: arrays on the device are global memory,
// a thread's own index is a register. Host arrays (the _h names) have no tier.
const tiers = (globals: string[], regs: string[]): Record<string, Tier> => ({
  ...Object.fromEntries(globals.map((n) => [n, 'global' as Tier])),
  ...Object.fromEntries(regs.map((n) => [n, 'reg' as Tier])),
});
const VEC_TIERS = tiers(['A', 'B', 'C', 'A_d', 'B_d', 'C_d'], ['i']);
const M0_TIERS = tiers(['X', 'Y', 'R', 'bias', 'x_d', 'y_d', 'r_d', 'bias_d', 'X_d', 'Y_d', 'R_d'], ['i']);
const HUNT_TIERS = tiers(['x', 'x_d'], ['i']);

// ---------------------------------------------------------------- reading the benchmark rows
const ms = (s: number) => `${fmt(s * 1000)} ms`;
/** A column of seconds shown in milliseconds; blank when the row lacks the key. */
const msOf = (head: string, key: string): Col => ({
  head,
  num: true,
  cell: (r) => {
    const v = num(r[key]);
    return v == null ? '' : fmt(v * 1000);
  },
});
/** The row called `name` with the largest n (and, when given, the sync setting). */
const largest = (s: BenchSet, name: string, sync?: number): BenchRow | undefined =>
  s.rows
    .filter((r) => r.name === name && (sync === undefined || r.sync === undefined || num(r.sync) === sync))
    .reduce<BenchRow | undefined>((best, r) => (best && (num(best.n) ?? 0) >= (num(r.n) ?? 0) ? best : r), undefined);
/** The host-array row a prediction compares against: on CUDA and HIP the one with the synchronise. */
const hostArrays = (s: BenchSet) => largest(s, 'phases', 1);

/** Lab 2.1: CPU loop time over one whole vecAdd call, at the largest size measured. */
const cpuOverOffload = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const r = largest(s, 'end_to_end');
    const v = num(r?.speedup_vs_cpu);
    const cpu = num(r?.cpu_s);
    const off = num(r?.offload_s);
    if (!r || v == null || !cpu || !off) return null;
    return { value: v, detail: `${ms(cpu)} for the CPU loop, ${ms(off)} for one whole call of vecAdd, at n = ${String(r.n)}` };
  });

/** Lab 2.2: the kernel's share of the host-array call. CUDA family: the kernel phase; Metal: the GPU time the command buffer reports. */
const kernelShare = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const r = hostArrays(s);
    const total = num(r?.total_s);
    if (!r || !total) return null;
    const kernel = num(r.kernel_s);
    if (kernel != null)
      return { value: (100 * kernel) / total, detail: `kernel phase ${ms(kernel)} of ${ms(total)} in all, with the synchronise, at n = ${String(r.n)}` };
    const gpu = num(r.gpu_s);
    if (gpu == null) return null;
    return { value: (100 * gpu) / total, detail: `${ms(gpu)} of GPU time reported by the command buffer, of ${ms(total)} in all, at n = ${String(r.n)}` };
  });

/** Lab 2.2: the host-array call's total over the resident path's time, at the largest size. */
const hostOverResident = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const host = hostArrays(s);
    const total = num(host?.total_s);
    const res = host ? benchRow(s, 'resident', { n: num(host.n) ?? -1 }) : undefined;
    const t = num(res?.seconds);
    if (!host || !total || !t) return null;
    return { value: total / t, detail: `${ms(total)} with host arrays in and out, ${ms(t)} on data already on the device, at n = ${String(host.n)}` };
  });

/** Lab 2.2, CUDA and HIP only: the kernel phase without the synchronise, as a percent of the one with it. */
const nosyncShare = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const r = largest(s, 'phases', 0);
    const v = num(r?.nosync_kernel_pct);
    if (!r || v == null || num(r.sync) !== 0) return null;
    return { value: v, detail: `kernel phase ${ms(num(r.kernel_s) ?? NaN)} without the synchronise, at n = ${String(r.n)}` };
  });

/** Lab 2.3: three round trips over one resident chain. */
const chainRatio = (reports: Reports) =>
  measure(reports, l3, (s) => {
    const r = benchRow(s, 'chain');
    const v = num(r?.ratio);
    const res = num(r?.resident_s);
    const rt = num(r?.roundtrips_s);
    if (!r || v == null || !res || !rt) return null;
    return { value: v, detail: `${ms(rt)} for three round trips, ${ms(res)} for one blockTail call, at ${String(r.rows)} × ${String(r.d)}` };
  });

// ---------------------------------------------------------------- lab 2.2: the columns depend on the backend
const PATH: Col = {
  head: 'Path',
  cell: (r) =>
    r.name === 'phases'
      ? r.sync === undefined
        ? 'host arrays in and out'
        : num(r.sync) === 1
          ? 'host arrays, sync after launch'
          : 'host arrays, no sync after launch'
      : r.name === 'resident'
        ? 'data already on the device'
        : r.name === 'cpu'
          ? 'CPU loop'
          : r.name,
};
const CUDA_PHASES: Col[] = [
  msOf('Allocate, ms', 'alloc_s'),
  msOf('Copy in, ms', 'h2d_s'),
  msOf('Kernel, ms', 'kernel_s'),
  msOf('Copy back, ms', 'd2h_s'),
  msOf('Free, ms', 'free_s'),
  numCol('Kernel, % of total', 'kernel_share_pct'),
  numCol('Kernel phase, % of the synchronised one', 'nosync_kernel_pct'),
];
const METAL_PHASES: Col[] = [
  msOf('Buffers and copy in, ms', 'wrap_s'),
  msOf('Encode and commit, ms', 'encode_s'),
  msOf('Wait, ms', 'wait_s'),
  msOf('Copy out, ms', 'readback_s'),
  msOf('GPU time, ms', 'gpu_s'),
];
const TOTAL: Col = {
  head: 'Total, ms',
  num: true,
  cell: (r) => {
    const v = num(r.name === 'phases' ? r.total_s : r.seconds);
    return v == null ? '' : fmt(v * 1000);
  },
};

function Lab2Results() {
  const { reports } = useLabs();
  const rows = Object.keys(reports).flatMap((b) => reports[b].results.filter((x) => x.lab === l2.id).flatMap((x) => x.bench));
  const cuda = rows.some((r) => r.alloc_s !== undefined);
  const metal = rows.some((r) => r.wrap_s !== undefined);
  return (
    <LabResults
      lab={l2}
      filter={(r) => r.name === 'phases' || r.name === 'resident' || r.name === 'cpu'}
      cols={[rawCol('n', 'n'), PATH, ...(cuda ? CUDA_PHASES : []), ...(metal ? METAL_PHASES : []), TOTAL]}
    />
  );
}

// ---------------------------------------------------------------- lab 2.4: what reported itself
// Backends that are not a GPU: the emulator and its race-detection flavour.
const EMULATED = new Set(['emu', 'race']);
interface HuntRun {
  backend: string;
  device: string;
  date: string;
  untouched: boolean;
  total: number;
  failed: number;
  /** Failing tests in which scalePrefix itself returned a non-zero status. */
  returned: number;
  /** The run was stopped before its end: a crash, or AddressSanitizer on the emulator. */
  stopped: boolean;
}
/** Per backend, how the last run of the reader's work/ for lab 2.4 failed. */
function huntRuns(reports: Reports): HuntRun[] {
  return Object.keys(reports).flatMap((b) => {
    const run = reports[b];
    const rs = run.results.filter((x) => x.lab === l4.id && x.variant === 'work' && x.built && x.test_rc != null);
    if (!rs.length) return [];
    const tests = rs.flatMap((x) => x.tests);
    const bad = tests.filter((t) => !t.pass);
    return [{
      backend: b,
      device: String(run.device.name ?? b),
      date: run.generated.slice(0, 10),
      untouched: run.work_state?.[l4.id] === 'untouched',
      total: tests.length,
      failed: bad.length,
      returned: bad.filter((t) => /^scalePrefix returned -?[1-9]/.test(t.detail ?? '')).length,
      stopped: rs.some((x) => x.test_rc !== 0 && x.test_rc !== 1),
    }];
  });
}
/** Failures that announced themselves: a non-zero status, or a stopped run. */
const selfReported = (r: HuntRun) => r.returned + (r.stopped ? 1 : 0);

function HuntReveal() {
  const { reports, loading, reload } = useLabs();
  const runs = huntRuns(reports);
  return (
    <>
      {runs.length ? (
        runs.map((r) => (
          <p key={r.backend} data-measured={selfReported(r)} data-backend={r.backend}>
            {EMULATED.has(r.backend) ? 'On the emulator' : `On ${r.device} (${r.backend})`}, in the run of {r.date},{' '}
            {r.untouched ? 'with work/ as shipped' : 'after work/ had been edited'}: {r.failed} of {r.total} tests failed.{' '}
            <b>{selfReported(r)}</b>{' '}
            {selfReported(r) === 1 ? 'failure announced itself' : 'failures announced themselves'}
            {r.returned ? `, ${r.returned} through a non-zero return value` : ''}
            {r.stopped ? `, ${EMULATED.has(r.backend) ? 'and the run was stopped by AddressSanitizer' : 'and the run stopped early'}` : ''}
            . {r.failed - r.returned > 0 ? `${r.failed - r.returned} were wrong values that only the comparison with the CPU reference found.` : ''}
          </p>
        ))
      ) : loading ? (
        <p data-measured="loading">Reading reports.</p>
      ) : (
        <p data-measured="none">
          No run of your work for this lab yet. Run <code>python3 labs/pmpp.py run 2.4 --backend emu</code>, then{' '}
          <code>python3 labs/pmpp.py run 2.4</code> on a machine with a GPU.{' '}
          <button className="quiet" onClick={reload}>
            Reload results
          </button>
        </p>
      )}
      <p className="muted">
        A test run counts failing tests, not defects, and one defect can fail many tests or hide another. Once all
        your tests pass, the last section of the lab's <code>README.md</code>, "After the hunt", says for each defect
        what each backend's documentation promises.
      </p>
    </>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS}>
      <LabsIntro>
        Four labs, each in its own directory under <code>labs/ch02/</code>. The first is the chapter's whole program,
        written by you from the first allocation to the last free. The second times it. The third builds the first
        milestone of the course project, and the fourth is a bug hunt in a host stub.
      </LabsIntro>
      <RunningLabs
        example="2.1"
        alsoShownIn={
          <>
            the measured chain of <a href={href('ch02', '2.3')}>section 2.3</a> and the phase times of{' '}
            <a href={href('ch02', '2.4')}>section 2.4</a>
          </>
        }
      />
      <BackendsAndEmulator />
      <p>
        None of these labs is built for <code>race</code>, the emulator's second flavour, which runs the threads
        concurrently under ThreadSanitizer. No thread in them touches an element that another thread writes.
      </p>
      <VerificationStatus />

      {/* ------------------------------------------------------------ 2.1 */}
      <h2>Lab 2.1: vector addition, end to end</h2>
      <p>
        You write the whole program of the chapter: the kernel of Fig. 2.10 and the host function of Fig. 2.13, which
        the driver calls with three ordinary host arrays, the way <code>main</code> calls it in Fig. 2.4. Allocate
        three device arrays, copy A and B in, compute the block count for the block size you are given, launch, check
        the launch, wait, copy C back, free, and return 0 or a non-zero status.
      </p>
      <Source section="2.3–2.6" pages="26–39" figs="Fig. 2.10, p. 35; Fig. 2.13, p. 38; the sidebar on p. 32" />
      <p>
        Every runtime call goes through a <code>CHECK</code> macro that you write once, from the error check on p. 32.
        That snippet does not compile as printed; <a href={href('ch02', '2.4')}>section 2.4</a> has you find out why.
        Your macro prints the error string, the file and the line, frees nothing, and makes the function return a
        non-zero value. It does not call <code>exit()</code>, because the driver tests what happens after a failure.
      </p>
      <LabFiles
        lab={l1}
        files={[
          ['kernel.cu', <><code>CHECK</code>, <code>vecAddKernel</code> and <code>vecAdd</code></>],
          ['kernel.metal', <><code>vec_add</code></>],
          ['launch.mm', <><code>vecAdd</code>: buffers, encoder, a dispatch of exactly n threads, commit, wait, check, copy out. <code>makeVecAddPipeline</code> is provided</>],
        ]}
      />
      <p>
        On Metal there are no error codes. You check every creation for nil, and the command buffer's status and
        error after the wait. The inputs reach their buffers through <code>newBufferWithBytes</code>, which copies,
        because the driver's arrays are ordinary heap memory and the copy-free wrapper needs page-aligned memory (
        <a href={href('ch02', '2.3')}>section 2.3</a>).
      </p>
      <Beyond>
        <p>The Metal kernel and host function are ours. The book's chapter is CUDA only.</p>
      </Beyond>
      <p>
        Run <code>python3 labs/pmpp.py run 2.1</code>, starting with <code>--backend emu</code>. The inputs lie
        between 0.5 and 1.5, so no correct sum is near 0, and <code>C_h</code> starts filled with NaN. The lengths sit
        on both sides of each block size: 1, 31, 32, 33, 255, 256, 257, 1000 and 4097, and on a device also 2²⁰ + 3.
        On CUDA, HIP and the emulator each length runs with 32 and with 256 threads per block, and one case uses the
        largest block the device reports. One more case on those three asks for a block twice as large as the device
        allows: it must
        return non-zero without ending the program, and the valid call made right after it must return 0 with the
        right sums.
      </p>
      <p>
        Read a failure by its shape. NaN left in <code>C_h</code> means nothing was copied back there. A wrong tail
        means too few blocks. An AddressSanitizer report on the emulator means an access outside an array, with the
        source line: a missing guard, a size in elements where bytes were meant, or a free before the copy back.
      </p>
      <LabPrediction
        id="ch02/labs/p-lab1-speedup"
        prompt={
          <>
            Before you run the benchmark. At the largest length it reaches, the time of the plain CPU loop of Fig. 2.4
            divided by the time of one whole call of your <code>vecAdd</code>, allocation, copies and free included,
            will be what? Above 1 means the GPU version wins.
          </>
        }
        unit="× as fast as the CPU loop"
        measure={cpuOverOffload}
        missing={
          <>
            Run <code>python3 labs/pmpp.py run 2.1 --bench</code> on a machine with a GPU. The emulator is never
            benchmarked.
          </>
        }
        reading={
          <>
            The kernel does one addition for every 12 bytes it moves, and the call also copies every one of those
            bytes across to the device and back. Section 2.3 counted that traffic; this number is what it costs against
            one CPU thread on your machine. Lab 2.2 takes the call apart.
          </>
        }
      />
      <LabResults
        lab={l1}
        filter={(r) => r.name === 'end_to_end'}
        cols={[
          rawCol('n', 'n'),
          msOf('CPU loop, ms', 'cpu_s'),
          msOf('Your vecAdd, whole call, ms', 'offload_s'),
          numCol('CPU time over vecAdd time', 'speedup_vs_cpu'),
        ]}
      />
      <ReferenceSolution lab={l1} tiers={VEC_TIERS} />

      {/* ------------------------------------------------------------ 2.2 */}
      <h2>Lab 2.2: where does the time go?</h2>
      <p>
        The book warns on p. 28 that this way of moving work to the device, copy in, compute, copy out, can be very
        inefficient, and that real programs keep large data on the device. It gives no numbers. Here you take them on
        your own machine. The starter holds a working <code>vecAdd</code> under a new name, the reference solution of
        lab 2.1 (paste in your own if you prefer it), and you add the clocks.
      </p>
      <Source section="2.3" pages="28" />
      <p>
        On CUDA, HIP and the emulator, <code>vecAddTimed</code> fills five phase times: allocation, copies in,
        kernel, copy back, free. Its flag <code>syncAfterLaunch</code> decides whether the kernel phase ends after a
        checked <code>cudaDeviceSynchronize()</code> or right after the launch statement, which is the experiment.
        A provided <code>launchVecAdd</code> runs the same kernel on arrays the driver already put on the device, and
        the driver times it with its own synchronise: the resident baseline. On Metal, <code>vecAddTimed</code> fills
        four host phases (buffers and copy in, encode and commit, wait, copy out) and the GPU time the command buffer
        reports, and you write <code>vecAddResident</code> for buffers the driver created in shared storage and filled
        through <code>contents</code>.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A3, A5, B1)">
        <p>
          A launch returns before its grid has run, so a timer around the launch alone measures only the launch.{' '}
          <code>cudaFree</code> "may perform implicit synchronization", and <code>hipFree</code> always synchronises
          first: a timer around a free can include waiting for the kernel.
        </p>
      </Today>
      <LabFiles
        lab={l2}
        files={[
          ['kernel.cu', <><code>vecAddTimed</code>; the kernel and <code>launchVecAdd</code> are provided and working</>],
          ['kernel.metal', <><code>vec_add</code>, provided complete</>],
          ['launch.mm', <><code>vecAddTimed</code> and <code>vecAddResident</code></>],
        ]}
      />
      <p>
        There is no <code>syncAfterLaunch</code> on Metal. On CUDA the blocking copy back waits for the kernel, so
        leaving the synchronise out only moves time from one phase to another. On Metal nothing waits for you, and
        reading <code>contents</code> before the wait has returned is an ordering error, not a timing choice.
      </p>
      <p>
        Run <code>python3 labs/pmpp.py run 2.2</code>. The tests check the sums for lengths 1, 257, 1000 and 4097 (and
        2²⁰ + 3 on a device) with both settings of the flag, then the phase times: each written, finite and not
        negative, and together no more than the driver's own clock around the call, with 5% and 50 microseconds of
        slack, and at least half of it. They check the resident path on arrays the driver owns. Whether a synchronise
        is in the right place is not a test. On the emulator every call is synchronous, so only the benchmark on a
        device can show it.
      </p>
      <LabPrediction
        id="ch02/labs/p-lab2-share"
        prompt={
          <>
            Before you run the benchmark. At the largest length, what percent of one host-array call is the kernel?
            On CUDA and HIP that is your kernel phase with the synchronise; on Metal, the GPU time the command buffer
            reports.
          </>
        }
        unit="percent of the call"
        measure={kernelShare}
        missing={
          <>
            Run <code>python3 labs/pmpp.py run 2.2 --bench</code> on a machine with a GPU.
          </>
        }
        reading={
          <>
            The rest of the call is allocation, copies, waiting and freeing: the part the book calls the cost of
            transparent outsourcing. The table has every phase, so you can see which one dominates at each length.
          </>
        }
      />
      <LabPrediction
        id="ch02/labs/p-lab2-resident"
        prompt={
          <>
            Before you run the benchmark. At the largest length, the time of one host-array call divided by the time
            of the same addition on data already on the device will be what?
          </>
        }
        unit="× the resident time"
        measure={hostOverResident}
        missing={
          <>
            Run <code>python3 labs/pmpp.py run 2.2 --bench</code> on a machine with a GPU.
          </>
        }
        reading={
          <>
            Both rows run the same kernel on the same lengths. What the resident path leaves out is everything that
            moves data between the caller's arrays and the device. On Metal it copies nothing at all: its row reports
            0 bytes copied, a count of what that code path does, not a timing.
          </>
        }
      />
      <LabPrediction
        id="ch02/labs/p-lab2-nosync"
        prompt={
          <>
            CUDA and HIP. With the synchronise after the launch removed, your kernel phase will read what percent of
            its synchronised value? Write down too which phase you expect to grow in its place.
          </>
        }
        unit="percent of the synchronised kernel phase"
        measure={nosyncShare}
        missing={
          <>
            This measurement exists only on CUDA and HIP. Metal has no synchronise to leave out, so its benchmark has
            no such row. Run <code>python3 labs/pmpp.py run 2.2 --bench</code> on a machine with CUDA or ROCm.
          </>
        }
        reading={
          <>
            Without the synchronise the clock stops when the launch statement returns. The kernel still runs, and its
            time turns up in a later phase that waits for it. Compare the two rows in the table phase by phase.
          </>
        }
      />
      <Lab2Results />
      <ReferenceSolution lab={l2} tiers={VEC_TIERS} />

      {/* ------------------------------------------------------------ 2.3 */}
      <h2>Lab 2.3: the elementwise operations a transformer needs</h2>
      <p>
        You write three kernels, each with a launch function beside it, and a host function that chains them on the
        device: a bias added to every row of a <code>rows × d</code> tensor, a scale and shift{' '}
        <code>alpha * x + beta</code>, and a residual added in place. Each kernel is the vector addition of Fig. 2.10
        with another body. The scale goes through <code>scaleShift</code>, a <code>__host__ __device__</code> helper
        that the compiler builds twice, and a provided host function calls the host build so the driver can compare
        the two. You have met this operation already: the kernel of the book's exercise 9, whose threads{' '}
        <a href={href('ch02', '2.6')}>section 2.6</a> counts, computes <code>b[i] = 2.7f * a[i] - 4.3f</code>, a
        scale and shift with the two constants written in. Here they become arguments.
      </p>
      <Source section="2.5" pages="35–36" figs="Fig. 2.10, Fig. 2.11; the affine map is exercise 9, Fig. 2.15, p. 42" />
      <p>
        This is milestone M0 of the course project, and it follows the conventions every later milestone composes
        with: row-major <code>float</code> tensors with the batch first, raw device pointers and explicit sizes as
        kernel arguments, and a <code>launch&lt;Name&gt;</code> function beside each kernel that computes its grid
        and launches, nothing more. Waiting and reading the error state are the caller's job, so that a caller can
        queue several launches on data that stays on the device. <code>blockTail</code> does exactly that:{' '}
        <code>y = alpha * (x + bias) + beta + r</code> with one set of copies in, three launches back to back, one
        check and one wait, and one copy out.
      </p>
      <Beyond>
        <p>
          The through-line project, <code>nanoinfer</code>, is ours, and so is using the chapter's kernel for bias,
          scale and residual. The book applies it to vector addition only, and uses the kernel of exercise 9 only to
          count threads.
        </p>
      </Beyond>
      <LabFiles
        lab={l3}
        files={[
          ['kernel.cu', <><code>scaleShift</code>, the three kernels, <code>launchAddBias</code>, <code>launchScaleShift</code>, <code>launchAddResidual</code> and <code>blockTail</code></>],
          ['kernel.metal', <>the helper <code>scale_shift</code> and the kernels <code>add_bias</code>, <code>scale_shift_k</code>, <code>add_residual</code></>],
          ['launch.mm', <><code>encodeAddBias</code>, <code>encodeScaleShift</code>, <code>encodeAddResidual</code>, and <code>blockTail</code> with all three dispatches in one command buffer</>],
        ]}
      />
      <p>
        On Metal there is no counterpart of the host build of <code>scaleShift</code>: every function in a{' '}
        <code>.metal</code> file is GPU code, and Apple documents no way to compile one definition for both sides (
        <a href={href('ch02', '2.7')}>section 2.7</a>). The helper in <code>kernel.metal</code> plays the part of a{' '}
        <code>__device__</code> function only.
      </p>
      <p>
        Run <code>python3 labs/pmpp.py run 2.3</code>. Each launch function is tested alone, on tensors the driver
        keeps on the device, at shapes 1 × 1, 1 × 257, 3 × 100, 4 × 256 and 7 × 33, and 64 × 4099 on a device. The
        bias differs from column to column and the shapes have fewer rows than columns, so a bias indexed by the row
        gives wrong values. Inputs must come back unchanged, outputs start as NaN, and on a device a block of marker
        values sits behind every tensor and must survive. On CUDA, HIP and the emulator the kernel's scale is also
        compared with <code>scaleShiftOnHost</code> element by element, within a tolerance: a compiler may fuse the
        multiply and the add into one operation that rounds once, and the two builds need not agree on that.
      </p>
      <LabPrediction
        id="ch02/labs/p-lab3-chain"
        prompt={
          <>
            Before you run the benchmark. At the size it picks, the time of the same three operations done as three
            separate round trips, each with its own copies in and out, divided by the time of one{' '}
            <code>blockTail</code> call, will be what? First count the tensor-sized copies each version makes: here
            the bias is one row and the scale is two numbers.
          </>
        }
        unit="× the time of one blockTail call"
        measure={chainRatio}
        missing={
          <>
            Run <code>python3 labs/pmpp.py run 2.3 --bench</code> on a machine with a GPU.
          </>
        }
        reading={
          <>
            The copy count says what each version moves; the ratio is what that costs on your device. On CUDA and HIP
            the round trips cross between host and device memory. On Metal nothing crosses a bus: the round trips copy
            host arrays into shared buffers and back, which is the cost of an interface that takes host arrays.
          </>
        }
      />
      <LabResults
        lab={l3}
        filter={(r) => r.name === 'op' || r.name === 'chain'}
        cols={[
          {
            head: 'Measured',
            cell: (r) => (r.name === 'chain' ? `blockTail, ${String(r.rows)} × ${String(r.d)}` : `${String(r.kernel ?? '')} alone, data on the device`),
          },
          numCol('Elements', 'elements', 8),
          {
            head: 'Time, ms',
            num: true,
            cell: (r) => {
              const v = num(r.name === 'chain' ? r.resident_s : r.seconds);
              return v == null ? '' : fmt(v * 1000);
            },
          },
          numCol('GB/s requested', 'gb_per_s'),
          msOf('Three round trips, ms', 'roundtrips_s'),
          numCol('Round trips over blockTail', 'ratio'),
        ]}
      />
      <ReferenceSolution lab={l3} tiers={M0_TIERS} />

      {/* ------------------------------------------------------------ 2.4 */}
      <h2>Lab 2.4: bug hunt, a stub that almost works</h2>
      <p>
        You are handed a complete program that compiles. <code>scalePrefix</code> multiplies the first{' '}
        <code>len</code> of <code>n</code> floats by <code>s</code> on the device and must leave the padding from{' '}
        <code>len</code> to <code>n</code> exactly as it was: the shape of an operation applied to the valid part of
        a padded batch. The CUDA version has four independent defects, the Metal version three. They are the mistakes
        this chapter's host stub invites, and finding them in code you did not write is the skill you will need for
        your own.
      </p>
      <Source section="2.4–2.6" pages="28–39" figs="Fig. 2.10, Fig. 2.13; the sidebar on p. 32" />
      <LabFiles
        lab={l4}
        files={[
          ['kernel.cu', <><code>scalePrefixKernel</code> and <code>scalePrefix</code>, with the four defects between them</>],
          ['kernel.metal', <><code>scale_prefix</code>, dispatched in whole threadgroups as llama.cpp's Metal backend does, so it computes its own index the way a CUDA kernel does</>],
          ['launch.mm', <><code>scalePrefix</code>, the Metal host function</>],
        ]}
      />
      <p>
        Each test fills the array with values between 1 and 2, calls <code>scalePrefix</code> with <code>s = 3</code>,
        and checks the whole array: the prefix scaled, the padding unchanged bit for bit. Arrays with padding come
        first, then arrays with none, then, on CUDA, HIP and the emulator, a launch the device cannot run, which must
        return non-zero.
      </p>
      <Predict
        id="ch02/labs/p-lab4-reported"
        prompt={
          <>
            Before you run anything on your GPU. Read <code>work/kernel.cu</code> and decide what the defects are. Of
            them, how many will the GPU report by itself, through an error code or a crash, as opposed to a wrong value
            that only a test against the CPU reference reveals?
          </>
        }
        unit="of the defects"
      >
        {() => <HuntReveal />}
      </Predict>
      <p>The method matters more than the fixes:</p>
      <ol>
        <li>
          Emulator first: <code>python3 labs/pmpp.py run 2.4 --backend emu</code>. It runs the kernel under
          AddressSanitizer, which stops at the first access outside a device array and names the line.
        </li>
        <li>
          Fix one thing and run again. The failures change as you go: one defect can hide another until it is fixed.
        </li>
        <li>
          Then run on your GPU, <code>python3 labs/pmpp.py run 2.4</code>, and note which defects it showed you by
          itself and which it hid.
        </li>
      </ol>
      <LabResults
        lab={l4}
        noBench="This lab has no benchmark. The result is the line above: every test passing on the emulator and on a device. As shipped, the program is meant to fail."
      />
      <ReferenceSolution lab={l4} tiers={HUNT_TIERS} />

      <h2>What these labs leave for later</h2>
      <p>
        <code>launchAddBias</code>, <code>launchScaleShift</code> and <code>launchAddResidual</code> are the pieces of
        milestone M0 that later milestones call on data that stays on the device. Chapter 3 adds the next milestone, a
        matrix multiplication for a linear layer, with the same conventions (<a href={href('ch03', 'labs')}>its
        labs</a>). The <a href={href('ch02', 'wrap')}>summary</a> has the chapter's mixed check and the list of places
        where the printed page differs from the tested code.
      </p>
    </LabsPage>
  );
}
