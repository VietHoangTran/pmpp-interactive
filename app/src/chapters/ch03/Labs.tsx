import type { Question } from '../../components/Ask';
import { Predict } from '../../components/Ask';
import { MATMUL_TIERS } from '../../components/Code';
import { Aside, Beyond, Source, Today, type Tier } from '../../components/notes';
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
} from '../../components/LabKit';
import { fmt, num, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 3's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch03', [
  { num: '3.1', dir: 'lab1_grayscale', name: 'grayscale' },
  { num: '3.2', dir: 'lab2_blur', name: 'blur' },
  { num: '3.3', dir: 'lab3_naive_gemm', name: 'naive GEMM (M1)' },
  { num: '3.4', dir: 'lab4_granularity', name: 'granularity' },
  { num: '3.5', dir: 'lab5_maphunt', name: 'map hunt' },
]);
const [l1, l2, l3, l4, l5] = LABS;

// Identifier colours for the reference solutions: buffers are global memory, everything
// a thread computes for itself is a register.
const reg = (...names: string[]): Record<string, Tier> => Object.fromEntries(names.map((n) => [n, 'reg' as Tier]));
const glob = (...names: string[]): Record<string, Tier> => Object.fromEntries(names.map((n) => [n, 'global' as Tier]));
const GRAY_TIERS = { ...glob('Pin', 'Pout'), ...reg('col', 'row', 'grayOffset', 'rgbOffset', 'r', 'g', 'b', 'gid') };
const BLUR_TIERS = { ...glob('in', 'out'), ...reg('col', 'row', 'pixVal', 'pixels', 'blurRow', 'blurCol', 'curRow', 'curCol', 'gid') };
const GEMM_TIERS = { ...MATMUL_TIERS, ...reg('gid', 'sum', 'Cvalue') };
const SEQ_TIERS = { ...glob('in', 'out'), ...reg('h', 't', 'b', 'sum', 'count', 'cur', 'd', 'gid') };

// ---------------------------------------------------------------- what each prediction is measured against
const seconds = (set: BenchSet, name: string, where: (r: BenchRow) => boolean) =>
  num(set.rows.find((r) => r.name === name && where(r))?.seconds);
const ms = (s: number) => `${fmt(s * 1000)} ms`;

/** Lab 3.1: time with blocks of 1 by 256 over time with blocks of 256 by 1, same picture. */
const tallOverWide = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const wide = seconds(s, 'grayscale', (r) => r.block_w === 256 && r.block_h === 1);
    const tall = seconds(s, 'grayscale', (r) => r.block_w === 1 && r.block_h === 256);
    if (!wide || !tall) return null;
    return { value: tall / wide, detail: `${ms(tall)} with blocks 1 wide and 256 high, ${ms(wide)} with blocks 256 wide and 1 high` };
  });
/** Lab 3.2: time at radius 4 over time at radius 1, same picture. */
const radiusCost = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const r1 = seconds(s, 'blur', (r) => r.radius === 1);
    const r4 = seconds(s, 'blur', (r) => r.radius === 4);
    if (!r1 || !r4) return null;
    return { value: r4 / r1, detail: `${ms(r4)} at r = 4, ${ms(r1)} at r = 1` };
  });
/** Lab 3.3: time at width 1024 over time at width 512. */
const doublingCost = (reports: Reports) =>
  measure(reports, l3, (s) => {
    const half = seconds(s, 'square', (r) => r.width === 512);
    const full = seconds(s, 'square', (r) => r.width === 1024);
    if (!half || !full) return null;
    return { value: full / half, detail: `${ms(full)} at width 1024, ${ms(half)} at width 512` };
  });
/** Lab 3.4: time of one thread per row over time of one thread per element, at the width the driver chose. */
const rowOverElement = (reports: Reports) =>
  measure(reports, l4, (s) => {
    const row = s.rows.find((r) => r.name === 'granularity' && r.mode === 'row');
    const element = seconds(s, 'granularity', (r) => r.mode === 'element');
    const perRow = num(row?.seconds);
    if (!row || !perRow || !element) return null;
    return { value: perRow / element, detail: `${ms(perRow)} per row, ${ms(element)} per element, at width ${String(row.width)}` };
  });

// ---------------------------------------------------------------- lab 3.5: which named tests failed
const NAMED = ['cube', 'wide', 'tall', 'odd batch', 'window'] as const;
// Backends that are not a GPU: the emulator and its race-detection flavour.
const EMULATED = new Set(['emu', 'race']);

interface NamedRun {
  backend: string;
  device: string;
  date: string;
  untouched: boolean;
  failed: string[];
  passed: string[];
}
/** Per device, how the five named tests of lab 3.5 went in the last run of the reader's work/. */
function namedRuns(reports: Reports): NamedRun[] {
  return Object.keys(reports)
    .filter((b) => !EMULATED.has(b))
    .flatMap((b) => {
      const run = reports[b];
      const tests = run.results.filter((x) => x.lab === l5.id && x.variant === 'work').flatMap((x) => x.tests);
      const of = (name: string) => tests.find((t) => t.case.startsWith(name + ':'));
      // A count means something only if all five were run.
      if (!NAMED.every((name) => of(name))) return [];
      return [{
        backend: b,
        device: String(run.device.name ?? b),
        date: run.generated.slice(0, 10),
        untouched: run.work_state?.[l5.id] === 'untouched',
        failed: NAMED.filter((name) => !of(name)!.pass),
        passed: NAMED.filter((name) => of(name)!.pass),
      }];
    });
}

function NamedReveal({ committed }: { committed: number }) {
  const { reports, loading, reload } = useLabs();
  const runs = namedRuns(reports);
  return (
    <>
      {runs.length ? (
        runs.map((r) => (
          <p key={r.backend} data-measured={r.failed.length} data-backend={r.backend}>
            On {r.device} ({r.backend}), in the run of {r.date},{' '}
            {r.untouched ? 'with work/ still as shipped' : 'after work/ had been edited'}: <b>{r.failed.length}</b> of
            the 5 named tests failed{r.failed.length ? ` (${r.failed.join(', ')})` : ''}
            {r.passed.length ? `; ${r.passed.join(', ')} passed` : ''}.{' '}
            {committed === r.failed.length
              ? 'That is the number you predicted. Check that the names match the ones you wrote down.'
              : 'That is not the number you predicted. Compare the names with the ones you wrote down.'}
          </p>
        ))
      ) : loading ? (
        <p data-measured="loading">Reading reports.</p>
      ) : (
        <p data-measured="none">
          No run of your work for this lab on a device yet. Run <code>python3 labs/pmpp.py run 3.5</code> on a machine
          with a GPU.{' '}
          <button className="quiet" onClick={reload}>
            Reload results
          </button>
        </p>
      )}
      <p className="muted">
        The count is from a device, where every test runs to the end. On the emulator a run stops at the first access
        outside a buffer, so the tests after that one are not run at all. Once you start fixing defects the count
        falls; what you predicted was the kernel as shipped.
      </p>
    </>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS}>
      <LabsIntro>
        Five labs, each in its own directory under <code>labs/ch03/</code>. The first three are the chapter's three
        kernels, written by you: grayscale, blur, and matrix multiplication for any shape. The fourth measures a
        design choice the book leaves as an exercise. The fifth is a bug hunt.
      </LabsIntro>
      <RunningLabs
        example="3.3"
        alsoShownIn={
          <>
            the ranking prediction of <a href={href('ch03', '3.4')}>section 3.4</a>
          </>
        }
      />
      <BackendsAndEmulator />
      <p>
        The emulator has a second flavour, <code>race</code>, which labs 3.3, 3.4 and 3.5 are also built for. It runs
        the emulated threads concurrently under ThreadSanitizer and reports two threads that touch the same element
        without synchronisation, with the source line. Ask for it with <code>--backend race</code>;{' '}
        <code>python3 labs/pmpp.py doctor</code> says whether your compiler can build it. The tables on this page
        show the first flavour only.
      </p>
      <VerificationStatus />

      {/* ------------------------------------------------------------ 3.1 */}
      <h2>Lab 3.1: colour to grayscale</h2>
      <p>
        You write the kernel of <a href={href('ch03', '3.2')}>section 3.2</a> and its launch: one thread per pixel, a
        2D grid of 2D blocks rounded up to whole blocks, the range test, and two flat offsets, one byte per pixel
        going out and three coming in. The grey level is L = 0.299 r + 0.587 g + 0.114 b, truncated to a byte. The
        block shape is an argument of the launch, so nothing in your code may assume 16.
      </p>
      <Source section="3.2" pages="48–55" figs="Fig. 3.4, p. 53; the launch is on p. 50" />
      <p>
        Do not copy Fig. 3.4. The lab's kernel differs from the printed one in four places: the cast, the constants,
        the argument order and the name. Section 3.2 has you find them, and the lab's <code>README.md</code> lists
        them. The tests find the cast and the constants for you if you do not.
      </p>
      <LabFiles
        lab={l1}
        files={[
          ['kernel.cu', <><code>colorToGrayscaleKernel</code> and <code>launchGrayscale</code></>],
          ['kernel.metal', <>the body of <code>grayscale</code>, over plain <code>uchar</code> buffers</>],
          ['launch.mm', <>the two <code>setBytes</code> calls and the dispatch in <code>encodeGrayscale</code></>],
        ]}
      />
      <p>
        The Metal launch uses <code>dispatchThreads</code>, so the grid is exactly the picture and you have to decide
        whether the kernel still needs its range test. A pixel is three separate bytes there too: a{' '}
        <code>uchar3</code> is four bytes in the Metal Shading Language.
      </p>
      <Beyond>
        <p>The Metal kernel and launch are ours. The book's chapter is CUDA only.</p>
      </Beyond>
      <p>
        Run <code>python3 labs/pmpp.py run 3.1</code>, starting with <code>--backend emu</code>. Your output is
        compared with a CPU reference, allowing one grey level either way, because the cast truncates and the order
        of three additions can move the byte by one. Flat pictures come first: pure red, green and blue must give 76,
        149 and 29, and white 255. Then a gradient with noise at 62 by 76 (the book's picture), 16 by 16, 1 by 1, 1
        by 300, 300 by 1, 33 by 17 and 64 by 80, and on a device 1500 by 2000, each with blocks of 16 by 16, 32 by 8
        and 8 by 32 threads. The output is filled with a marker value first, so a pixel you never write is counted.
        On a device the buffers are larger than the picture and the bytes after it must come back unchanged.
      </p>
      <p>
        The benchmark then times one large picture under several block shapes. In two of them a block holds 256
        threads: a run of pixels along a row in one, a run down a column in the other. NVIDIA's guide says that "the
        use of multi-dimensional thread blocks and grids is for convenience only and does not affect performance",
        which is about writing one mapping in one, two or three dimensions. The same passage goes on to number a
        block's threads x fastest and adds: "This affects how threads are assigned to warps", a word chapter 4
        explains. Neither sentence tells you what these two shapes will do.
      </p>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, A2)">
        <p>Both quotations are from NVIDIA's CUDA programming guide.</p>
      </Today>
      <LabPrediction
        id="ch03/labs/p-lab1-tall-wide"
        prompt={
          <>
            Before you run the benchmark. It times one large picture with several block shapes. Blocks 256 wide and 1
            high, and blocks 1 wide and 256 high, launch the same number of threads, and every thread does the same
            work. The time with the tall blocks divided by the time with the wide ones will be what? (1 means the
            same.)
          </>
        }
        unit="× the wide-block time"
        measure={tallOverWide}
        missing={
          <>
            Run <code>python3 labs/pmpp.py run 3.1 --bench</code> on a machine with a GPU. The benchmark skips a block
            shape the device does not allow, and without both shapes there is no ratio.
          </>
        }
        reading={
          <>
            Both launches do the same arithmetic on the same bytes. What differs is which pixels the threads of one
            block hold: a run along a row, or a run down a column. Why that can matter is the subject of chapters 4
            and 6; for now the number is an observation to carry there.
          </>
        }
      />
      <LabResults
        lab={l1}
        filter={(r) => r.name === 'grayscale'}
        cols={[
          { head: 'Block, wide × high', cell: (r) => `${String(r.block_w)} × ${String(r.block_h)}` },
          { head: 'Picture, rows × columns', cell: (r) => `${String(r.height)} × ${String(r.width)}` },
          msCol,
          numCol('Megapixels per second', 'mpix_per_s'),
        ]}
      />
      <ReferenceSolution lab={l1} tiers={GRAY_TIERS} />

      {/* ------------------------------------------------------------ 3.2 */}
      <h2>Lab 3.2: image blur</h2>
      <p>
        You write the blur kernel of <a href={href('ch03', '3.3')}>section 3.3</a> and its launch. The mapping is lab
        3.1's, but now each thread loops: it averages the input pixels in a square patch around its own position and
        leaves out the part of the patch that hangs over an edge. Two tests, a running sum, a count, and a division
        by the count.
      </p>
      <Source section="3.3" pages="55–59" figs="Fig. 3.8, p. 57" />
      <p>
        One difference from the page: the book fixes the patch radius with a constant <code>BLUR_SIZE</code>. Here
        the radius is a kernel argument <code>r</code>, so the benchmark can change it. The patch is 2r + 1 pixels on
        a side. The book prints no launch for this kernel; yours is the grayscale launch with 16 by 16 blocks.
      </p>
      <LabFiles
        lab={l2}
        files={[
          ['kernel.cu', <><code>blurKernel</code> and <code>launchBlur</code></>],
          ['kernel.metal', <>the body of <code>blur</code>. The position Metal hands you is unsigned; the offsets inside a patch are not</>],
          ['launch.mm', <>the dispatch in <code>encodeBlur</code>, with <code>dispatchThreadgroups</code>, which counts groups and so rounds up like a CUDA grid</>],
        ]}
      />
      <p>
        Run <code>python3 labs/pmpp.py run 3.2</code>. The comparison again allows one grey level. The shapes are 62
        by 76 with r = 1, 2 and 3, a single pixel, a single row and a single column, 3 by 3 with r = 3 where the patch
        is larger than the picture, 33 by 17 with r = 7, and on a device 1500 by 2000. A failure is reported as two
        counts, wrong border pixels and wrong interior pixels. A wrong border with a correct interior points at the
        inner test or at what you divide by; a wrong interior points at the mapping.
      </p>
      <p>
        The first test hands the kernel the top 62 rows of a 66-row buffer. A read that runs past the right edge of a
        row lands on the start of the next row, which is valid memory, so no bounds checker objects and only the
        values can show it. In a buffer of exact size the same mistake is caught at the last row alone.
      </p>
      <LabPrediction
        id="ch03/labs/p-lab2-radius"
        prompt={
          <>
            Before you run the benchmark. It blurs one large picture at r = 1, 2, 4 and 8. Going from r = 1 to r = 4,
            each thread reads 81 values where it read 9. The time at r = 4 divided by the time at r = 1 will be what?
          </>
        }
        unit="× the time at r = 1"
        measure={radiusCost}
        missing={
          <>
            Run <code>python3 labs/pmpp.py run 3.2 --bench</code> on a machine with a GPU.
          </>
        }
        reading={
          <>
            Nine times the reads per thread is a statement about what the kernel asks for. The ratio is what your
            device made of it. Look at r = 2 and r = 8 in the table as well before you explain the number, and run it
            twice: on a busy machine these timings move.
          </>
        }
      />
      <LabResults
        lab={l2}
        filter={(r) => r.name === 'blur'}
        cols={[
          rawCol('r', 'radius'),
          {
            head: 'Values per patch',
            num: true,
            cell: (r) => {
              const radius = num(r.radius);
              return radius == null ? '' : String((2 * radius + 1) ** 2);
            },
          },
          { head: 'Picture, rows × columns', cell: (r) => `${String(r.height)} × ${String(r.width)}` },
          msCol,
          numCol('Megapixels per second', 'mpix_per_s'),
        ]}
      />
      <ReferenceSolution lab={l2} tiers={BLUR_TIERS} />

      {/* ------------------------------------------------------------ 3.3 */}
      <h2>Lab 3.3: naive GEMM for a linear layer</h2>
      <p>
        You write the kernel of Fig. 3.11 for <code>C[j×l] = A[j×k] × B[k×l]</code>: row-major <code>float</code>, any
        sizes, one thread per element of C, 16 by 16 blocks. The figure has one <code>Width</code>. You have three
        sizes, and every <code>Width</code> in the figure becomes exactly one of them. The grid's x covers l and its
        y covers j.
      </p>
      <Source section="3.4" pages="59–64" figs="Fig. 3.11, p. 61; the rectangular definition is on p. 60" />
      <p>
        This is the first half of milestone M1 of the through-line project. As a linear layer, j is the batch, which
        comes first, k the number of inputs and l the number of outputs, and B holds the weight as (in, out). That is
        the transpose of what PyTorch stores, (out, in), and of what ggml multiplies by; both read their two operands
        along rows, where this kernel reads its second operand down a column. The lab keeps the book's convention
        because <a href={href('ch05', 'labs')}>lab 5.3</a> keeps it too: same argument order, same storage, so the
        tiled kernel replaces this one later without touching the caller. One of the tests builds a weight stored
        PyTorch's way, transposes it on the host and checks <code>y = x Wᵀ</code>.
      </p>
      <Beyond>
        <p>
          The through-line project, <code>nanoinfer</code>, is ours, and so are the rectangular kernel and its launch:
          the book's kernel is square only and it prints no launch for it.
        </p>
      </Beyond>
      <LabFiles
        lab={l3}
        files={[
          ['kernel.cu', <><code>naiveGemmKernel</code> and <code>launchNaiveGemm</code>, both empty in the starter</>],
          ['kernel.metal', <>the body of <code>naive_gemm</code></>],
          ['launch.mm', <>the dispatch at the end of <code>encodeNaiveGemm</code></>],
        ]}
      />
      <p>
        On Metal the kernel takes <code>[[thread_position_in_grid]]</code>, the launch rounds up to whole 16 by 16
        threadgroups with <code>dispatchThreadgroups</code>, and the kernel keeps its guard. PyTorch's MPS backend
        launches its plain Metal matmul kernel the same way, and llama.cpp's Metal backend dispatches whole
        threadgroups everywhere. MLX calls <code>dispatchThreads</code>, which launches exactly the grid you ask for and needs no
        guard. Either passes the tests.
      </p>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, C3, D3, D4, D5)">
        <p>
          PyTorch rounds up with 16 × 16 groups and tests the range before its store. llama.cpp's kernels bound a
          strided loop where this lab has a range test. MLX's one-element-per-thread kernels have no guard.
        </p>
      </Today>
      <p>
        Run <code>python3 labs/pmpp.py run 3.3</code>. The shapes are square at 4 (the size of the book's Fig. 3.12),
        16, 33 and 64, then 5 × 7 × 3, a single row (one token through a layer), a single output column (matrix times
        vector, book exercise 2), an inner dimension of 1 (an outer product), 17 × 40 × 33, the PyTorch-layout case,
        and on a device two larger ones. C is filled with a marker value first, so an element you never write is
        reported, and on a device a value stored past the end of C is reported too. With <code>--bench</code>, square
        widths 256, 512 and 1024 are timed, then the 1024 → 4096 layer of lab 5.3 at batch 1, 16 and 256.
      </p>
      <LabPrediction
        id="ch03/labs/p-lab3-doubling"
        prompt={
          <>
            Before you run the benchmark. Doubling the width from 512 to 1024 multiplies the arithmetic by 8. The
            kernel's time at width 1024 divided by its time at width 512 will be what?
          </>
        }
        unit="× the time at width 512"
        measure={doublingCost}
        missing={
          <>
            Run <code>python3 labs/pmpp.py run 3.3 --bench</code> on a machine with a GPU.
          </>
        }
        reading={
          <>
            Eight is what the operation count predicts. Whatever you measured, the GFLOP/s column of the table is the
            baseline that chapter 5 starts from: lab 5.3 times the same layer with the same row names, so the two
            tables can be read side by side.
          </>
        }
      />
      <LabResults
        lab={l3}
        filter={(r) => r.name === 'square' || r.name === 'linear_layer'}
        cols={[
          {
            head: 'Shape, j × k × l',
            cell: (r) =>
              r.name === 'square'
                ? `${String(r.width)} × ${String(r.width)} × ${String(r.width)}`
                : `${String(r.batch)} × ${String(r.k)} × ${String(r.l)} (layer, batch ${String(r.batch)})`,
          },
          msCol,
          numCol('GFLOP/s', 'gflops'),
          numCol('Rows per second', 'rows_per_s'),
          numCol('Time over half the width', 'time_vs_half_width'),
        ]}
      />
      <ReferenceSolution lab={l3} tiers={GEMM_TIERS} />

      {/* ------------------------------------------------------------ 3.4 */}
      <h2>Lab 3.4: one thread per element, per row or per column</h2>
      <p>
        Fig. 3.11 gives every element of P its own thread. The book's first exercise asks for two coarser designs: one
        thread computes a whole output row, and one thread computes a whole output column. You write both, each on a
        1D grid of 256-thread blocks with a guard, for square matrices as in the exercise. The per-element kernel and
        its launch are provided in the same file.
      </p>
      <Source section="3.5" pages="64" figs="exercises 1a and 1b; the kernel is Fig. 3.11, p. 61" />
      <LabFiles
        lab={l4}
        files={[
          ['kernel.cu', <><code>rowKernel</code> with <code>launchRow</code>, and <code>colKernel</code> with <code>launchCol</code></>],
          ['kernel.metal', <><code>row_matmul</code> and <code>col_matmul</code>, each taking a scalar <code>[[thread_position_in_grid]]</code></>],
          ['launch.mm', <>the 1D dispatches in <code>encodeRow</code> and <code>encodeCol</code></>],
        ]}
      />
      <p>
        Run <code>python3 labs/pmpp.py run 3.4</code>. Each of the three kernels is tested and reported on its own, at
        widths 1, 4, 16, 33 and 100, and on a device also 257 and 512. As shipped, the provided kernel passes and your
        two fail. Most of these widths leave part of a 256-thread block without work, which is what the guard is for.
      </p>
      <p>
        With <code>--bench</code>, all three kernels are timed at one width. The driver picks it: it doubles the width
        until the per-element kernel runs long enough to time, and stops earlier if one more doubling would push the
        slowest kernel past half a second a run. It prints the width it chose and why.
      </p>
      <LabPrediction
        id="ch03/labs/p-lab4-row"
        prompt={
          <>
            Before you run the benchmark. <a href={href('ch03', '3.4')}>Section 3.4</a> asked you to rank the three
            kernels. Now commit to a factor: at the benchmark's width, the time of one thread per row divided by the
            time of one thread per element will be what? All three do the same arithmetic. They differ in how many
            threads share it and in which direction each thread writes.
          </>
        }
        unit="× the per-element time"
        measure={rowOverElement}
        missing={
          <>
            Run <code>python3 labs/pmpp.py run 3.4 --bench</code> on a machine with a GPU.
          </>
        }
        reading={
          <>
            The table has the number of threads with work beside each time. Exercise 1c asks for the pros and cons of
            each design; write yours now that you have numbers, and note the width they were taken at, because the
            ratio is not a constant of the device.
          </>
        }
      />
      <LabResults
        lab={l4}
        filter={(r) => r.name === 'granularity'}
        cols={[
          {
            head: 'One thread per',
            cell: (r) => (r.mode === 'element' ? 'element (provided)' : r.mode === 'row' ? 'row' : r.mode === 'col' ? 'column' : String(r.mode ?? '')),
          },
          rawCol('Width', 'width'),
          numCol('Threads with work', 'threads', 7),
          msCol,
          numCol('GFLOP/s', 'gflops'),
          numCol('Time over per-element', 'time_vs_element'),
        ]}
      />
      <ReferenceSolution lab={l4} tiers={GEMM_TIERS} />

      {/* ------------------------------------------------------------ 3.5 */}
      <h2>Lab 3.5: map hunt</h2>
      <p>
        You are handed one kernel and its launch, in CUDA and in Metal, with four independent defects planted in them.
        It compiles, and it passes its first test. These are the mistakes this chapter's index arithmetic invites, and
        finding them in code you did not write is the skill you will need for your own.
      </p>
      <Source section="3.2" pages="55" figs="the 3D index form; the clipped loop is Fig. 3.8, §3.3, p. 57" />
      <p>
        The kernel takes a windowed mean along the sequence axis of a <code>[batch, seq, hidden]</code> tensor:{' '}
        <code>out[b][t][h]</code> is the mean of <code>in[b][s][h]</code> over s from t − r to t + r, for the s that
        exist. Hidden is on x, seq on y, batch on z, in blocks of 8 × 8 × 4 threads.
      </p>
      <Beyond>
        <p>
          The kernel is ours, built from two things that are the book's: the 3D mapping of p. 55 and the blur
          kernel's clipped loop with its counter, kept to one axis.
        </p>
      </Beyond>
      <LabFiles
        lab={l5}
        files={[
          ['kernel.cu', <><code>seqMeanKernel</code> and <code>launchSeqMean</code>, with the four defects between them</>],
          ['kernel.metal', <><code>seq_mean</code>, with the same defects</>],
          ['launch.mm', <><code>encodeSeqMean</code>, which has the launch's defect</>],
        ]}
      />
      <p>
        The first five tests are named. The first is a cube, and each of the next four moves away from it in one
        direction:
      </p>
      <Aside>
        <p>
          The tests named <code>mixed</code> after the five are mixtures of sizes. They pass once all four defects are
          gone.
        </p>
      </Aside>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-table="maphunt-tests">
          <thead>
            <tr><th>Test</th><th className="num">batch</th><th className="num">seq</th><th className="num">hidden</th><th className="num">r</th></tr>
          </thead>
          <tbody>
            <tr><td><code>cube</code></td><td className="num">4</td><td className="num">16</td><td className="num">16</td><td className="num">0</td></tr>
            <tr><td><code>wide</code></td><td className="num">4</td><td className="num">6</td><td className="num">20</td><td className="num">0</td></tr>
            <tr><td><code>tall</code></td><td className="num">4</td><td className="num">20</td><td className="num">6</td><td className="num">0</td></tr>
            <tr><td><code>odd batch</code></td><td className="num">5</td><td className="num">16</td><td className="num">16</td><td className="num">0</td></tr>
            <tr><td><code>window</code></td><td className="num">4</td><td className="num">16</td><td className="num">16</td><td className="num">1</td></tr>
          </tbody>
        </table>
      </div>
      <Predict
        id="ch03/labs/p-lab5-failing"
        prompt={
          <>
            Before the first run. Read the kernel and the launch in <code>work/kernel.cu</code>, and write down which
            of the five named tests you expect to fail on a device with the code as shipped. How many is that?
          </>
        }
        unit="of the 5 named tests"
      >
        {(v) => <NamedReveal committed={v} />}
      </Predict>
      <p>The method matters more than the fixes:</p>
      <ol>
        <li>
          Emulator first: <code>python3 labs/pmpp.py run 3.5 --backend emu</code>. Fix one thing and run again. The
          failures change as you go.
        </li>
        <li>
          Know what each kind of failure looks like there. An access outside the tensor stops the run with an
          AddressSanitizer report: read whether it says READ or WRITE, the first frame naming a line of{' '}
          <code>kernel.cu</code>, and how many bytes before or after the buffer the address is. An access to the wrong
          element inside the tensor is a valid address, so the sanitizer is silent and you get wrong values, with a
          count of elements no thread wrote. A mistake in the launch is ordinary host code that computes a few
          integers; no tool flags it, and you see only its effect.
        </li>
        <li>
          Each test runs under three thread orders. A correct kernel gives the same output under all three. If a test
          reports different wrong values from one order to the next, two threads are writing the same element.{' '}
          <code>--backend race</code> then names the line.
        </li>
        <li>
          Use the shapes. A test that fails moved away from the cube in one direction, and that direction is what the
          defect needs in order to show.
        </li>
        <li>
          Make the same fixes in the Metal pair and run on the device. There the driver puts guard elements before and
          after each tensor, so a stray write lands in memory the test owns and is counted. For each defect, note what
          the device showed you: a wrong value, an element never written, or only a write into the guard elements. A
          real program has no guard elements, and a defect of the last kind leaves every result you check correct.
        </li>
      </ol>
      <LabResults
        lab={l5}
        noBench="This lab has no benchmark. The result is the line above: every test passing on the emulator and on a device. As shipped, the kernel is meant to fail."
      />
      <ReferenceSolution lab={l5} tiers={SEQ_TIERS} />

      <h2>What these labs leave for later</h2>
      <p>
        Lab 3.3 is half of M1. Its 16 by 16 block is a fixed choice; chapter 4 completes the milestone by choosing the
        launch configuration from what the device reports. Chapter 5 then replaces the kernel behind the same
        arguments with a tiled one (<a href={href('ch05', 'labs')}>lab 5.3</a>, milestone M2) and measures the two
        against each other. The <a href={href('ch03', 'wrap')}>summary</a> has the chapter's mixed check and the list
        of places where the printed page differs from the tested code.
      </p>
    </LabsPage>
  );
}
