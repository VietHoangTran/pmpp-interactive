import type { Question } from '../../components/Ask';
import { MATMUL_TIERS } from '../../components/Code';
import { Beyond, Source, Today, type Tier } from '../../components/notes';
import {
  BackendsAndEmulator,
  LabFiles,
  LabPrediction,
  LabResults,
  LabsIntro,
  LabsPage,
  ReferenceSolution,
  RaceStatus,
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
import { fmt, labStatus, num, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 6's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch06', [
  { num: '6.1', dir: 'lab1_coalescing', name: 'coalescing' },
  { num: '6.2', dir: 'lab2_vector', name: 'vector loads' },
  { num: '6.3', dir: 'lab3_banks', name: 'bank conflicts' },
  { num: '6.4', dir: 'lab4_linear', name: 'linear layer (M2)' },
  { num: '6.5', dir: 'lab5_doublebuf', name: 'double buffering' },
]);
const [l1, l2, l3, l4, l5] = LABS;

// Identifier colours per lab: names mean different things in different labs (`in` is a
// buffer in lab 6.1 and a size in lab 6.4), so there is no page-wide map. Lab 6.2 has
// none: its buffers are x, y and z, and colouring those would colour every .x field.
const TIERS_61: Record<string, Tier> = { in: 'global', out: 'global', i: 'reg', row: 'reg', col: 'reg' };
const TIERS_63: Record<string, Tier> = { in: 'global', out: 'global', tile: 'shared', tx: 'reg', ty: 'reg', r: 'reg', c: 'reg', outRow: 'reg', outCol: 'reg' };
const TIERS_64: Record<string, Tier> = {
  X: 'global', W: 'global', Y: 'global', Xs: 'shared', Ws: 'shared',
  Yvalue: 'reg', tx: 'reg', ty: 'reg', Row: 'reg', Col: 'reg', ph: 'reg', xk: 'reg', wo: 'reg', wk: 'reg', i: 'reg',
};

// ---------------------------------------------------------------- reading the benchmark rows
// Several labs tell their rows apart by a string field (`variant` is the kernel's name),
// which LabKit's benchRow() does not match, so these helpers do.
const rowWhere = (set: BenchSet | undefined, name: string, where: Record<string, string | number>) =>
  set?.rows.find((r) => r.name === name && Object.entries(where).every(([k, v]) => r[k] === v));
const ms = (r?: BenchRow) => (num(r?.seconds) != null ? `${fmt(num(r!.seconds)! * 1000)} ms` : 'no time');

/** Lab 6.1: the strided read at stride 2, as a percentage of stride 1. */
const stride2 = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const two = rowWhere(s, 'strided', { stride: 2 });
    const one = rowWhere(s, 'strided', { stride: 1 });
    const v = num(two?.relative_to_stride1);
    if (!two || v == null) return null;
    return {
      value: v * 100,
      detail: `${fmt(num(two.gb_per_s) ?? NaN)} GB/s at stride 2 against ${fmt(num(one?.gb_per_s) ?? NaN)} GB/s at stride 1`,
    };
  });
/** Lab 6.2: the float4 kernel's speed-up over one float per thread. */
const vec4Speedup = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const r = rowWhere(s, 'vecadd', { variant: 'vec4' });
    const v = num(r?.speedup_vs_scalar);
    if (!r || v == null) return null;
    return { value: v, detail: `${ms(r)} against ${ms(rowWhere(s, 'vecadd', { variant: 'scalar' }))} at n = ${String(r.n)}` };
  });
/** Lab 6.3: the bank probe's time per access at stride 32, over stride 1. */
const probe32 = (reports: Reports) =>
  measure(reports, l3, (s) => {
    const r = rowWhere(s, 'bank_probe', { stride: 32 });
    const v = num(r?.relative_to_stride1);
    if (!r || v == null) return null;
    const one = rowWhere(s, 'bank_probe', { stride: 1 });
    return {
      value: v,
      detail: `${fmt(num(r.ns_per_access) ?? NaN)} ns per warp-wide access at stride 32, ${fmt(num(one?.ns_per_access) ?? NaN)} at stride 1, ${String(r.lanes)} lanes`,
    };
  });
/** Lab 6.3: transpose time without padding over time with it, at T = 32. */
const padGain = (reports: Reports) =>
  measure(reports, l3, (s) => {
    const plain = rowWhere(s, 'transpose', { tile: 32, pitch: 32 });
    const padded = rowWhere(s, 'transpose', { tile: 32, pitch: 33 });
    const a = num(plain?.seconds);
    const b = num(padded?.seconds);
    if (!a || !b) return null;
    return { value: a / b, detail: `${ms(plain)} at pitch 32, ${ms(padded)} at pitch 33, on a ${String(plain!.rows)} × ${String(plain!.cols)} matrix` };
  });
/** Lab 6.4: the corner-turned, padded kernel over the provided baseline, batch 256, T = 16. */
const linearSpeedup = (reports: Reports) =>
  measure(reports, l4, (s) => {
    const r = rowWhere(s, 'linear', { variant: 'tuned', tile: 16, batch: 256 });
    const v = num(r?.speedup_vs_baseline);
    if (!r || v == null) return null;
    return {
      value: v,
      detail: `${fmt(num(r.gflops) ?? NaN)} GFLOP/s against the baseline's ${fmt(num(rowWhere(s, 'linear', { variant: 'baseline', tile: 16, batch: 256 })?.gflops) ?? NaN)}, ${String(r.in)} → ${String(r.out)} layer`,
    };
  });
/** Lab 6.5: one barrier per phase (double buffered) over two, at T = 16. */
const oneBarrier = (reports: Reports) =>
  measure(reports, l5, (s) => {
    const r = rowWhere(s, 'matmul', { variant: 'double_buffered', tile: 16 });
    const v = num(r?.speedup_vs_two_barriers);
    if (!r || v == null) return null;
    return { value: v, detail: `${ms(r)} against ${ms(rowWhere(s, 'matmul', { variant: 'two_barriers', tile: 16 }))} at width ${String(r.width)}` };
  });

// ---------------------------------------------------------------- results columns
// Labs 6.1 and 6.3 record several kinds of row (a probe, a copy, the kernel itself, the
// SIMD width), so their tables say in the first column what each row measured.
const what61: Col = {
  head: 'Measurement',
  cell: (r) =>
    r.name === 'copy'
      ? `copy (provided), ${String(r.n)} floats`
      : r.name === 'strided'
        ? `strided read, stride ${String(r.stride)}`
        : r.name === 'transpose'
          ? `naive transpose, ${r.kernel === 'write_rows' || r.variant === 'write_rows' ? 'rows written (strided loads)' : 'rows read (strided stores)'}, ${String(r.rows)} × ${String(r.cols)}`
          : r.name === 'simd_width'
            ? 'SIMD width the probe used'
            : r.name,
};
const what63: Col = {
  head: 'Measurement',
  cell: (r) =>
    r.name === 'bank_probe'
      ? `bank probe, stride ${String(r.stride)}`
      : r.name === 'copy'
        ? `copy (provided), ${String(r.n)} floats`
        : r.name === 'transpose'
          ? `tiled transpose, T = ${String(r.tile)}, pitch ${String(r.pitch)}, ${String(r.rows)} × ${String(r.cols)}`
          : r.name === 'simd_width'
            ? 'SIMD width the probe used'
            : r.name,
};
const resultCol: Col = {
  head: 'Result',
  num: true,
  cell: (r) =>
    r.name === 'simd_width'
      ? String(r.width ?? '')
      : r.name === 'bank_probe'
        ? `${fmt(num(r.ns_per_access) ?? NaN)} ns per access`
        : num(r.gb_per_s) != null
          ? `${fmt(num(r.gb_per_s)!)} GB/s`
          : '',
};
const vsStride1: Col = {
  head: 'Relative to stride 1',
  num: true,
  cell: (r) => {
    const v = num(r.relative_to_stride1);
    if (v == null) return '';
    return r.name === 'bank_probe' ? `${fmt(v)} × the time` : `${fmt(v)} of the bandwidth`;
  },
};
const msOrBlank: Col = { ...msCol, cell: (r) => (r.name === 'simd_width' ? '' : msCol.cell(r)) };

// ---------------------------------------------------------------- milestone M2
function MilestoneStatus() {
  const { reports, loading } = useLabs();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const pass = backends.filter((b) => labStatus(reports[b], l4.id, 'work') === 'pass');
  const name = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  return (
    <p data-milestone={loading ? 'loading' : pass.join(',') || 'none'}>
      {loading
        ? 'Reading reports.'
        : !backends.length
          ? 'No lab results are recorded on this machine yet.'
          : pass.length
            ? <>Your own <code>linearKernel</code> passes on {pass.map(name).join(' and ')} on this machine: the layer is in the framework layout there. Whether it is tuned is the benchmark&rsquo;s to say.</>
            : <>Your own <code>linearKernel</code> does not pass on any backend of this machine yet.</>}
    </p>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS}>
      <LabsIntro>
        Five labs, each in its own directory under <code>labs/ch06/</code>. Three are measurements of a rule you can
        state but not predict on your device: coalescing, vector loads, bank conflicts. The fourth moves the
        project&rsquo;s linear layer to the weight layout frameworks use and tunes it, milestone M2. The fifth removes a
        barrier from the tiled kernel and has a race detector check the result. Code you do not edit, the baselines and
        the probes, sits in each lab&rsquo;s <code>provided/</code> folder.
      </LabsIntro>
      <RunningLabs
        example="6.1"
        alsoShownIn={
          <>
            sections <a href={href('ch06', '6.1')}>6.1</a>, <a href={href('ch06', '6.3')}>6.3</a>,{' '}
            <a href={href('ch06', '6.4')}>6.4</a>, <a href={href('ch06', '6.7')}>6.7</a> and{' '}
            <a href={href('ch06', '6.9')}>6.9</a>
          </>
        }
      />
      <BackendsAndEmulator />
      <p>
        The emulator has a second flavour, <code>race</code>, which labs 6.3, 6.4 and 6.5 are also built for. It runs the emulated
        threads concurrently under ThreadSanitizer and reports two threads that touch the same element without
        synchronisation, with both source lines. Ask for it with <code>--backend race</code>;{' '}
        <code>python3 labs/pmpp.py doctor</code> says whether your compiler can build it. The tables on this page show
        the first flavour only.
      </p>
      <p>
        Every cost this chapter talks about, transactions, bank passes, barriers, exists only on a GPU. The emulators
        check that each kernel computes the right values and touches no memory it should not, and nothing about its
        speed. Each lab therefore asks you to predict before you benchmark, and the prediction is compared with your
        device, never with the book&rsquo;s.
      </p>
      <VerificationStatus />

      {/* ------------------------------------------------------------ 6.1 */}
      <h2>Lab 6.1: coalescing on your device</h2>
      <p>
        You write three kernels that differ only in which address each thread touches. A strided read: thread{' '}
        <code>i</code> copies <code>in[i * stride]</code> to <code>out[i]</code>, so the write is contiguous at every
        stride and the read only at stride 1. Then two naive transposes of a rows × cols matrix, one that reads rows and
        writes columns, one that reads columns and writes rows. A provided copy, which reads and writes contiguously, is
        the yardstick. None of these is a listing in the book: they are probes of the rule section 6.1 states, and on a
        Mac they are the only way to learn what that rule looks like there.
      </p>
      <Source section="6.1" pages="124–132" figs="Figs. 6.2 and 6.3; stores, p. 130" />
      <LabFiles
        lab={l1}
        files={[
          ['kernel.cu', <><code>stridedReadKernel</code>, <code>transposeReadRowsKernel</code>, <code>transposeWriteRowsKernel</code> and their three launches</>],
          ['kernel.metal', <><code>strided_read</code>, <code>transpose_read_rows</code> and <code>transpose_write_rows</code>, indexing with <code>int</code></>],
          ['launch.mm', <>the dispatches in <code>encodeStridedRead</code>, <code>encodeTransposeReadRows</code> and <code>encodeTransposeWriteRows</code></>],
        ]}
      />
      <p>
        Run <code>python3 labs/pmpp.py run 6.1</code>, starting with <code>--backend emu</code>. The strided read is
        tested at n = 1, 31, 32, 33 and 1000 with strides 1, 2, 3, 8 and 32, and on a device also at n = 65536 with
        stride 64. Its input holds exactly <code>n * stride</code> floats, so on the emulator AddressSanitizer names the
        line of any read past it. The transposes run on 1 × 1, 1 × 37, 37 × 1, 32 × 32, 33 × 65 and 100 × 7, and on a
        device on 1000 × 513 and 2048 × 2048. Outputs start as NaN, so an element you never write fails, and the
        comparison is exact: these kernels move values and compute nothing. With <code>--bench</code> the copy, the
        strided read at strides 1 to 64 over one input of 2<sup>25</sup> floats, and both transposes on a 4096 × 4096
        matrix are timed, and the SIMD width your device reports is recorded.
      </p>
      <LabPrediction
        id="ch06/labs/p-lab1-stride2"
        prompt={
          <>
            Before you run the benchmark. The figure the lab prints for the strided read is useful bytes (read plus
            written) per second, relative to stride 1. At stride 2, what percentage of stride 1 will you measure? On
            NVIDIA a stride-2 load uses half of every 32-byte sector it fetches, and the write is contiguous at every
            stride: start from there, then decide what your caches might change.
          </>
        }
        unit="% of stride 1"
        measure={stride2}
        reading={
          <>
            If time followed the bytes moved, NVIDIA&rsquo;s sector model would put this figure at two thirds, not
            one half: stride 1 moves 8 bytes per element for 8 used, stride 2 moves 12 for 8 used, because the load
            fetches every sector and the store does not change. A measurement near that says your device fetches whole
            units of about that size and nothing on chip rescues the unused half. Apple documents no unit at all, so
            on a Mac this number is the only description of the rule you have. Read the rest of the sweep in the table:
            where does the curve stop falling, and at what stride?
          </>
        }
      />
      <p>
        The second prediction for this lab, which naive transpose is faster, is in{' '}
        <a href={href('ch06', '6.1')}>section 6.1</a> under &ldquo;Stores count too&rdquo;, where the book&rsquo;s
        claim about stores is made. It reads the same benchmark rows.
      </p>
      <LabResults lab={l1} cols={[what61, msOrBlank, resultCol, vsStride1]} />
      <ReferenceSolution lab={l1} tiers={TIERS_61} />

      {/* ------------------------------------------------------------ 6.2 */}
      <h2>Lab 6.2: vector loads, with the boundaries done right</h2>
      <p>
        Figure 6.12 makes each thread of vector addition add four pairs of floats with one 16-byte load per input and
        one 16-byte store, and leaves the boundaries to Exercise 4: as printed it reads past the end whenever the length
        is not a multiple of four. Section 6.5 then calls the same kernel an example of thread coarsening. This lab
        separates the two ideas. You write two kernels in which each thread owns the same four elements: one moves
        them with one <code>float4</code> per array (four elements per instruction), the other with section 6.5&rsquo;s
        coarsening loop and one float per access (four elements per thread). A provided kernel with one float per
        thread is the baseline.
      </p>
      <Source section="6.3" pages="138–139" figs="Fig. 6.12; Exercise 4, p. 155; coarsening, §6.5 pp. 141–143" />
      <LabFiles
        lab={l2}
        files={[
          ['kernel.cu', <><code>vecAdd4Kernel</code>, <code>vecAddCoarseKernel</code> and their launches</>],
          ['kernel.metal', <><code>vec_add4</code>, which sees each buffer as <code>float4</code> for full chunks and as <code>float</code> for the tail, and <code>vec_add_coarse</code></>],
          ['launch.mm', <><code>encodeVecAdd4</code>, which binds each buffer twice at two indices, and <code>encodeVecAddCoarse</code></>],
        ]}
      />
      <p>
        Run <code>python3 labs/pmpp.py run 6.2</code>, starting with <code>--backend emu</code>. Both kernels run for n =
        1, 2, 3, 4, 5, 7, 8, 9, 255, 256, 257, 1023, 1024 and 1025, every remainder of n modulo 4, and on a device also
        for 2<sup>20</sup>, 2<sup>20</sup> + 3 and 10,000,001. Each array holds exactly n floats, so on the emulator a{' '}
        <code>float4</code> that reaches past the end is an AddressSanitizer error naming the line; on a GPU the same
        mistake may go unnoticed. The output starts as NaN, so a skipped tail fails. The pointers are the allocations
        themselves, 16-byte aligned: the emulator cannot tell a misaligned <code>float4</code> from an aligned one, so
        no test passes <code>x + 1</code>. With <code>--bench</code> the three kernels are timed at one size, grown from
        2<sup>22</sup> until the baseline takes over 5 ms, up to 2<sup>26</sup>.
      </p>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md A11, B5, C3)">
        <p>
          The cast needs a pointer aligned to 16 bytes. NVIDIA&rsquo;s blog shows it, not its guides; AMD&rsquo;s docs do
          not describe it. On Metal the documented route is to declare the buffer <code>device const float4*</code>,
          which is what <code>vec_add4</code> does.
        </p>
      </Today>
      <LabPrediction
        id="ch06/labs/p-lab2-vec4"
        prompt={
          <>
            Before you run the benchmark. The book&rsquo;s argument is 75% fewer load instructions for the same data
            (p. 139). The one-float kernel is already coalesced, runs at full occupancy and moves 12 bytes per
            addition. What speed-up of the <code>float4</code> kernel over it will you measure at the largest size?
          </>
        }
        unit="× the one-float kernel"
        measure={vec4Speedup}
        reading={
          <>
            Fewer instructions save time only if instructions were what the kernel waited on. A kernel that already
            keeps the memory system busy has little to gain from issuing its requests in fewer, wider pieces; the book
            names the case where it pays, a kernel that cannot reach full occupancy (p. 139). Then compare the
            coarsened kernel in the table: same four elements per thread, one float per access. Apply section
            6.1&rsquo;s rule to one of its load instructions before you explain its number. Section{' '}
            <a href={href('ch06', '6.3')}>6.3</a> reads the same rows.
          </>
        }
      />
      <LabResults
        lab={l2}
        filter={(r) => r.name === 'vecadd'}
        cols={[
          {
            head: 'Kernel measured',
            cell: (r) =>
              r.variant === 'scalar' ? 'one float per thread (provided)' : r.variant === 'vec4' ? 'float4, four per instruction' : r.variant === 'coarse4' ? 'coarsened, four per thread' : String(r.variant),
          },
          rawCol('n', 'n'),
          msCol,
          numCol('GB/s (12 bytes per element)', 'gb_per_s'),
          numCol('Speed-up over one float', 'speedup_vs_scalar'),
        ]}
      />
      <ReferenceSolution lab={l2} />

      {/* ------------------------------------------------------------ 6.3 */}
      <h2>Lab 6.3: provoke a bank conflict, then pad it away</h2>
      <p>
        Section 6.4&rsquo;s example is the corner-turning store: the threads of a warp write down a column of a 32 × 32
        shared tile and every one of them lands in bank 0. The book shows the two declarations and nothing else. Here
        you write a kernel whose whole job is that turn, a matrix transpose through a shared tile: read the input tile
        row by row, one barrier, write the output tile row by row, which means reading the shared tile down a column.
        The tile&rsquo;s pitch is a run-time argument, so one kernel serves both the unpadded layout (pitch T) and the
        padded one (pitch T + 1). Two instruments are provided: a probe in which one warp reads and writes a shared
        array at Exercise 5&rsquo;s strides, and a copy.
      </p>
      <Source section="6.4" pages="139–141" figs="Exercise 5, p. 155; turning a tile, Fig. 6.4" />
      <LabFiles
        lab={l3}
        files={[
          ['kernel.cu', <><code>tiledTransposeKernel</code> and <code>launchTiledTranspose</code>; the shared tile is declared for you</>],
          ['kernel.metal', <><code>tiled_transpose</code>, with the tile width given at run time</>],
          ['launch.mm', <>the threadgroups in <code>encodeTiledTranspose</code></>],
        ]}
      />
      <p>
        First work Exercise 5 by hand (section <a href={href('ch06', '6.4')}>6.4</a> has the table): the probe&rsquo;s
        rows are in the same order. Then run <code>python3 labs/pmpp.py run 6.3</code>, starting with{' '}
        <code>--backend emu</code>. Your transpose runs on 1 × 1, 7 × 5, 16 × 16, 17 × 33, 64 × 48, 100 × 1 and 1 × 100,
        and on a device on 1000 × 1000 and 513 × 2049, each at both pitches, against an output that starts as NaN, with
        an exact comparison. On the emulator a missing barrier fails in at least one of the three thread orders, and{' '}
        <code>--backend race</code> names the two lines it leaves unordered. The
        tests cannot see which pitch you used: the output is the same either way, and only the benchmark can tell. With{' '}
        <code>--bench</code> the probe runs strides 1, 2, 4, 7, 8, 12, 16, 24, 31, 32 and 33, and the transpose is timed
        at T = 16 and 32 at both pitches beside the copy.
      </p>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md A12, B4, C2)">
        <p>
          NVIDIA: 32 banks of 4 bytes, and lanes reading one 32-bit word do not conflict. AMD: 32 or 64 banks serving 16
          lanes per cycle. Apple documents no banks, and advises against many threads reading one location.
        </p>
      </Today>
      <LabPrediction
        id="ch06/labs/p-lab3-stride32"
        prompt={
          <>
            Before you run the benchmark. In NVIDIA&rsquo;s model a warp at stride 32 needs 32 passes where stride 1
            needs one. The probe times a chain of dependent accesses by one warp, so latency and the loop around each
            access are in every number. How many times longer per access will your device take at stride 32 than at
            stride 1?
          </>
        }
        unit="× the time at stride 1"
        measure={probe32}
        reading={
          <>
            The 32 counts passes through the banks; the probe measures everything else as well, so even on NVIDIA the
            ratio is diluted by the fixed cost of each access. Look at the shape of the whole sweep rather than one
            row: do strides that share a factor with 32 (2, 4, 8, 16, 24) stand out from the odd ones (7, 31, 33)? On
            AMD, remember the 16 lanes per cycle; on a Mac, no document says what to expect, and this is the evidence.
          </>
        }
      />
      <LabPrediction
        id="ch06/labs/p-lab3-pad"
        prompt={
          <>
            And for the transpose at T = 32, the book&rsquo;s case: its time at pitch 32 divided by its time at pitch 33.
            (Above 1 means the padding made it faster.)
          </>
        }
        unit="× the padded time"
        measure={padGain}
        missing={
          <>
            This needs the T = 32 transpose rows. Run <code>python3 labs/pmpp.py run 6.3 --bench</code>; on Metal they
            appear only where the pipeline accepts 1024 threads per threadgroup.
          </>
        }
        reading={
          <>
            The conflict is in one shared-memory read per element, and the transpose also reads and writes global
            memory once per element. Compare both pitches with the copy in the table: a transpose that already runs at
            about copy speed has nothing left for its layout to win. Then do the T = 16 pair, after working out what a
            32-lane warp touches when the block is 16 wide.
          </>
        }
      />
      <LabResults lab={l3} cols={[what63, msOrBlank, resultCol, vsStride1]} />
      <ReferenceSolution lab={l3} tiers={TIERS_63} />

      {/* ------------------------------------------------------------ 6.4 */}
      <h2>Lab 6.4: the linear layer in the framework layout, corner-turned (milestone M2)</h2>
      <p>
        Lab 5.3 computed <code>C = A B</code> with B stored (k, l). Frameworks do not store a linear layer that way:
        PyTorch&rsquo;s <code>nn.Linear</code> keeps its weight as (out_features, in_features) and computes{' '}
        <code>y = x W^T</code>, and llama.cpp&rsquo;s <code>ggml_mul_mat</code> uses the same layout. Read the lab 5.3 way,
        that <code>W^T</code> is the book&rsquo;s column-major N (p. 128): neighbouring threads fetch floats{' '}
        <code>in</code> apart.
      </p>
      <Beyond>
        <p>
          The weight layout is from PyTorch&rsquo;s and llama.cpp&rsquo;s source (<code>notes/today/ch03-facts.md</code>{' '}
          D6). The kernel is our solution to Exercise 2; the book gives no listing for it.
        </p>
      </Beyond>
      <p>
        You write <code>linearKernel(X, W, Y, batch, in, out)</code> computing <code>Y = X W^T</code> with two changes
        to your lab 5.3 kernel, both in the W tile. Load it corner-turned, so neighbouring <code>threadIdx.x</code> read
        neighbouring floats of one row of W. And pad the shared tile to <code>[TILE_WIDTH][TILE_WIDTH + 1]</code>,
        because the turned store writes down a column. This is the book&rsquo;s Exercise 2 in the layout the project
        needs.
      </p>
      <Source section="6.1" pages="129–130" figs="Fig. 6.4; padding, §6.4 pp. 139–141; Exercise 2, p. 155" />
      <LabFiles
        lab={l4}
        files={[
          ['kernel.cu', <>the W tile&rsquo;s declaration, its corner-turned load and the product line in <code>linearKernel</code></>],
          ['kernel.metal', <>the same three places in <code>linear</code></>],
        ]}
      />
      <p>
        <code>work/launch.mm</code> is complete. Run <code>python3 labs/pmpp.py run 6.4</code>, starting with{' '}
        <code>--backend emu</code>. The shapes (batch, in, out) are 3 × 3 × 3, 1 × 1 × 1, 16 × 16 × 16, 32 × 48 × 16,
        17 × 33 × 9, one token through a 40 → 7 layer, 5 × 1 × 9 and 33 × 65 × 50, and on a device also 4 × 300 × 1000,
        257 × 129 × 511 and one token through a 1024 → 4096 layer. The output starts filled with a marker value and is
        compared with a CPU reference within a tolerance. The tests check results only: the provided baseline, the
        same tiled kernel reading <code>W[Col * in + k]</code> uncoalesced, passes them too, and so does a kernel that
        forgets the padding. Whether your load coalesces shows in the benchmark: with <code>--bench</code> the baseline
        and your kernel take turns on lab 5.3&rsquo;s 1024 → 4096 layer at batch 1, 16, 64, 256 and 1024, at T = 16
        and 32.
      </p>
      <LabPrediction
        id="ch06/labs/p-lab4-speedup"
        prompt={
          <>
            Before you run the benchmark. At T = 32 one warp&rsquo;s W load touches 32 sectors in the baseline and 4
            corner-turned, in NVIDIA&rsquo;s model; at T = 16 a 32-thread warp covers two rows of the block, so work that
            figure out first. The W load is one part of each phase. What speed-up of your kernel over the baseline will
            you measure at batch 256, T = 16?
          </>
        }
        unit="× the baseline"
        measure={linearSpeedup}
        reading={
          <>
            Lab 5.2 showed caches serving traffic the model charged for. Here, in the baseline, the other warps of the
            same block need the rest of every sector a warp fetches, and they ask for it moments later. The model
            counts sectors per instruction; the device may serve the block&rsquo;s whole tile from a handful of lines
            either way. Compare the batch sizes and both tile widths in the table before you explain the number, then
            remove the padding column, run <code>--bench</code> again, and see what it was buying.
          </>
        }
      />
      <LabResults
        lab={l4}
        filter={(r) => r.name === 'linear'}
        cols={[
          { head: 'Layer kernel', cell: (r) => (r.variant === 'baseline' ? 'baseline (provided)' : r.variant === 'tuned' ? 'corner-turned, padded' : String(r.variant)) },
          rawCol('T', 'tile'),
          rawCol('Batch', 'batch'),
          msCol,
          numCol('GFLOP/s', 'gflops'),
          numCol('Speed-up over baseline', 'speedup_vs_baseline'),
        ]}
      />
      <ReferenceSolution lab={l4} tiers={TIERS_64} />
      <h3>The project: milestone M2</h3>
      <p>
        From here on the project&rsquo;s weights are stored (out, in) row-major, as <code>nn.Linear</code> and ggml
        store them; tensors are row-major <code>float</code> with the batch first; kernels take raw device pointers and
        explicit sizes. <code>launchLinear(X, W, Y, batch, in, out)</code> is the signature chapter 15 builds on, where
        the kernel is coarsened and register-tiled into milestone M3.
      </p>
      <MilestoneStatus />

      {/* ------------------------------------------------------------ 6.5 */}
      <h2>Lab 6.5: double buffering, with a race detector watching</h2>
      <p>
        You are handed lab 5.2&rsquo;s tiled matrix multiplication with its second barrier deleted. It compiles, and it
        is wrong: a thread that finishes its dot product loads the next phase&rsquo;s tile over slots its neighbours
        have not read yet. You make it correct with one barrier per phase, without putting the second barrier back:
        give each operand two tiles, let the phases alternate between the two pairs, keep the barrier after the loads
        and none after the dot product. The provided baseline is lab 5.2&rsquo;s two-barrier kernel.
      </p>
      <Source section="6.7" pages="146–147" figs="applied to Fig. 5.9, p. 108" />
      <p>
        Lab 5.2&rsquo;s loop writes first and reads second, so the barrier that guards the false (write-after-read)
        dependence is its second one. In the book&rsquo;s toy loop the read comes first and it is the first one, and
        p. 147 says &ldquo;second&rdquo; where it means that first (see the{' '}
        <a href={href('ch06', 'wrap')}>summary</a>). Section <a href={href('ch06', '6.7')}>6.7</a> steps through both.
      </p>
      <LabFiles
        lab={l5}
        files={[
          ['kernel.cu', <>the tile declarations and the two places that choose a pair in <code>doubleBufferedMatmulKernel</code>, then TODO 4, a comment saying why it is correct</>],
          ['kernel.metal', <>the same change in <code>double_buffered_matmul</code></>],
        ]}
      />
      <p>The method matters more than the fix:</p>
      <ol>
        <li>
          Before you change anything, run the starter under ThreadSanitizer:{' '}
          <code>python3 labs/pmpp.py run 6.5 --backend race</code>. Its report names a store into a tile and the
          multiply-add that reads the tiles. Write down which thread does what, in which order, for that report to be
          possible.
        </li>
        <li>
          <code>python3 labs/pmpp.py run 6.5 --backend emu</code>. The starter passes at width 16 and fails at 32 and 64
          under all three thread orders. Explain the width 16 case before you go on.
        </li>
        <li>
          Fix <code>work/kernel.cu</code>, and run both emulator flavours until both are clean. Then make the same change
          in <code>work/kernel.metal</code> and run on every backend you have. On a GPU each case is launched 10 times
          with the output refilled with NaN before each launch.
        </li>
      </ol>
      <p>
        The tests compare with a CPU reference at widths 16, 32 and 64 (T = 16) on the emulator, 32 and 48 on the race
        backend, and 16, 32, 256 and 33 × T on a device. Three other mistakes were each checked to fail on their own: two
        pairs of tiles with no barrier at all, the barrier after the dot product instead of after the loads, and the dot
        product reading the other pair. A clean run is evidence, not proof: the race backend sees only the schedules that
        happened, and ten launches are ten schedules. The argument in TODO 4 is what shows the kernel correct.
      </p>
      <RaceStatus lab={l5} />
      <LabPrediction
        id="ch06/labs/p-lab5-onebarrier"
        prompt={
          <>
            Before you run the benchmark. One barrier per phase instead of two, at the price of twice the shared memory
            per block. At T = 16, what speed-up over the two-barrier kernel will you measure? Weigh one barrier against
            the 16 multiply-adds each thread does between barriers.
          </>
        }
        unit="× the two-barrier kernel"
        measure={oneBarrier}
        reading={
          <>
            A barrier costs what the slowest thread of the block makes the others wait, and a phase of 16 multiply-adds
            per thread is short. The second pair of tiles is not free either: chapter 5 showed that shared memory per
            block can limit how many blocks are resident. MLX&rsquo;s production Metal GEMM keeps two barriers per step,
            the book&rsquo;s &ldquo;before&rdquo; loop. Whatever you measured is a fact about this kernel on your device,
            not a rule about double buffering.
          </>
        }
      />
      <LabResults
        lab={l5}
        filter={(r) => r.name === 'matmul'}
        cols={[
          { head: 'Kernel measured', cell: (r) => (r.variant === 'two_barriers' ? 'two barriers (provided)' : r.variant === 'double_buffered' ? 'double buffered, one barrier' : String(r.variant)) },
          rawCol('T', 'tile'),
          rawCol('Width', 'width'),
          msCol,
          numCol('GFLOP/s', 'gflops'),
          numCol('Speed-up over two barriers', 'speedup_vs_two_barriers'),
        ]}
      />
      <ReferenceSolution lab={l5} tiers={MATMUL_TIERS} />
      <p>
        Real kernels take the idea one step further: while the threads compute on one pair of tiles, the next pair is
        already on its way. That needs a copy the threads do not wait for, and it is where chapter 15 picks the idea up.
      </p>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md A16, B8, C7, D3)">
        <p>
          NVIDIA documents double buffering today mostly with asynchronous copies into shared memory. AMD documents
          such a copy only for CDNA 3 and 4. Metal documents none, so on a Mac the synchronous form of this lab is the
          one you have. MLX&rsquo;s Metal GEMM keeps two barriers per step.
        </p>
      </Today>
    </LabsPage>
  );
}
