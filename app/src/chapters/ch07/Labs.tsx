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
  type Col,
} from '../../components/LabKit';
import { fmt, num, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 7's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch07', [
  { num: '7.1', dir: 'lab1_constconv', name: 'constant filter' },
  { num: '7.2', dir: 'lab2_tiledconv', name: 'tiled with halos' },
]);
const [l1, l2] = LABS;
// Lab 5.1 measured each device's bandwidth: the model ceilings of lab 7.1 are built on it.
const [l51] = defineLabs('ch05', [{ num: '5.1', dir: 'lab1_roofline', name: 'roofline' }]);

// Identifier colours per lab. In lab 7.1 `F` is a global pointer in the basic kernel and a
// constant buffer in Metal's conv_const, so it is left uncoloured; `F_c` and the by-value
// struct `f` are constant memory (CUDA passes kernel arguments through it, ch07-facts A4).
const REG: Record<string, Tier> = {
  Pvalue: 'reg', outRow: 'reg', outCol: 'reg', inRow: 'reg', inCol: 'reg', fRow: 'reg', fCol: 'reg',
  row: 'reg', col: 'reg', tx: 'reg', ty: 'reg', tileRow: 'reg', tileCol: 'reg',
};
const TIERS_71: Record<string, Tier> = { ...REG, N: 'global', P: 'global', F_c: 'const', f: 'const' };
const TIERS_72: Record<string, Tier> = { ...REG, N: 'global', P: 'global', F_c: 'const', F: 'const', N_s: 'shared' };

// ---------------------------------------------------------------- what each prediction is measured against
/** Lab 7.1: the constant-memory kernel's speed-up over the basic one at r = 2. */
const constSpeedup = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const r = benchRow(s, 'const', { radius: 2 });
    const v = num(r?.speedup_vs_basic);
    if (!r || !v) return null;
    return {
      value: v,
      detail: `${fmt(num(r.gflops) ?? NaN)} GFLOP/s against ${fmt(num(benchRow(s, 'basic', { radius: 2 })?.gflops) ?? NaN)} for the basic kernel, ${String(r.width)} × ${String(r.height)} image`,
    };
  });
/** Lab 7.1: the by-value struct over the constant buffer, at r = 2. */
const byValueOverConst = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const a = num(benchRow(s, 'byvalue', { radius: 2 })?.gflops);
    const b = num(benchRow(s, 'const', { radius: 2 })?.gflops);
    if (!a || !b) return null;
    return { value: a / b, detail: `${fmt(a)} GFLOP/s by value against ${fmt(b)} GFLOP/s through the constant buffer or symbol` };
  });
/** Lab 7.2: the halo-tiled kernel's speed-up over the constant-memory kernel, 32-wide tile, r = 2. */
const haloSpeedup = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const r = benchRow(s, 'tiled_halo', { tile: 32, radius: 2 });
    const v = num(r?.speedup_vs_const);
    if (!r || !v) return null;
    return {
      value: v,
      detail: `${fmt(num(r.gflops) ?? NaN)} GFLOP/s tiled; the formula's traffic ratio there is ${fmt(num(r.model_traffic_ratio) ?? NaN)}`,
    };
  });
/** Lab 7.2: the cached-halo kernel over the halo-tiled one, 32-wide tile, r = 2. */
const cachedOverHalo = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const c = num(benchRow(s, 'tiled_cached', { tile: 32, radius: 2 })?.gflops);
    const h = num(benchRow(s, 'tiled_halo', { tile: 32, radius: 2 })?.gflops);
    if (!c || !h) return null;
    return { value: c / h, detail: `${fmt(c)} GFLOP/s cached against ${fmt(h)} GFLOP/s with the halo in shared memory` };
  });

// ---------------------------------------------------------------- results tables
const image: Col = { head: 'Image', num: true, cell: (r) => (r.width != null ? `${String(r.width)} × ${String(r.height)}` : '') };
const blank = (v: unknown) => (v == null ? '' : String(v));

/** Lab 7.1's rows, with each kernel set against the ceiling its model gives it on the same device. */
function Lab71Results() {
  const { reports } = useLabs();
  // Lab 5.1's bandwidth, per backend, from the reader's own run where it passed.
  const bandwidth = new Map(
    preferredBench(reports, l51).flatMap((s) => {
      const v = num(benchRow(s, 'roofline')?.bandwidth_gb_per_s);
      return v ? [[s.backend, v] as const] : [];
    }),
  );
  const ceiling = (r: BenchRow) => {
    const bw = bandwidth.get(String(r.backend));
    const ai = num(r.model_intensity);
    return bw && ai ? ai * bw : undefined;
  };
  return (
    <LabResults
      lab={l1}
      cols={[
        {
          head: 'Variant',
          cell: (r) =>
            r.name === 'pipeline'
              ? `conv_const pipeline: execution width ${blank(r.threadExecutionWidth)}, at most ${blank(r.maxTotalThreadsPerThreadgroup)} threads`
              : String(r.name),
        },
        rawCol('r', 'radius'),
        image,
        msCol,
        numCol('GFLOP/s', 'gflops'),
        numCol('Speed-up over basic', 'speedup_vs_basic'),
        numCol('Model FLOP/B', 'model_intensity'),
        {
          head: '× model ceiling',
          num: true,
          cell: (r) => {
            const c = ceiling(r);
            const g = num(r.gflops);
            return c && g ? fmt(g / c) : '';
          },
        },
      ]}
    />
  );
}

const KERNEL_72: Record<string, string> = { const: 'constant memory (provided)', tiled_halo: 'halo tiles (part A)', tiled_cached: 'cached halo (part B)' };

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS}>
      <LabsIntro>
        Two labs, each in its own directory under <code>labs/ch07/</code>. The first writes the basic convolution three
        times, changing only where the filter lives. The second tiles it: you write the halo kernel of section 7.5 and
        repair the cached-halo kernel of section 7.6, then measure both against the formula.
      </LabsIntro>
      <RunningLabs
        example="7.1"
        alsoShownIn={
          <>
            the ceiling panel of <a href={href('ch07', '7.3')}>section 7.3</a>, the speed-up plot of{' '}
            <a href={href('ch07', '7.5')}>section 7.5</a> and the timing grid of <a href={href('ch07', '7.6')}>section 7.6</a>
          </>
        }
      />
      <BackendsAndEmulator />
      <VerificationStatus />

      {/* ------------------------------------------------------------ 7.1 */}
      <h2>Lab 7.1: the filter in constant memory, three ways</h2>
      <p>
        You write the one-thread-per-output kernel of section 7.2 three times, and the only thing that changes is where
        the filter lives. First as a pointer to global memory, which is Fig. 7.7 made to compile and given an output
        guard. Then in a <code>__constant__</code> array that the launch fills with <code>cudaMemcpyToSymbol</code>, which
        is Fig. 7.9. Then in a struct passed by value with the launch, the alternative the book names on p. 172. All
        three apply the filter without flipping it, use zero ghost cells and take a run-time radius from 0 to 7.
      </p>
      <Source section="7.2–7.4" pages="164–172" figs="Figs. 7.7 and 7.9" />
      <p>
        The point is a measurement. The model says the second and third kernels need half the global traffic of the
        first, 0.5 FLOP/B against 0.25, so they should run twice as fast if memory decides. Section 7.3 explained why
        that need not happen: the model charges every read to DRAM, and your device&rsquo;s caches may already be
        serving the filter.
      </p>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <>
              <code>convBasicKernel</code>, <code>convConstKernel</code> with <code>F_c</code>, and{' '}
              <code>convByValueKernel</code>, each with its <code>launch</code> function. Keep the{' '}
              <code>__constant__</code> array, its kernel and the copy in this file: an <code>extern __constant__</code>{' '}
              declaration in another file, as the book suggests on p. 169, is an error in nvcc&rsquo;s default mode (ch07-facts A1).
            </>,
          ],
          ['kernel.metal', <><code>conv_basic</code>, <code>conv_const</code> and <code>conv_by_value</code></>],
          ['launch.mm', <><code>encodeConvBasic</code>, <code>encodeConvConst</code> and <code>encodeConvByValue</code></>],
        ]}
      />
      <p>
        Metal has no symbol to copy into: a program-scope <code>constant</code> array has to be initialised in the shader
        source. So on Metal the three ways are three kinds of kernel argument: a <code>device const float*</code>{' '}
        buffer, a <code>constant float*</code> buffer, and a <code>constant Filter&amp;</code> that the host fills with{' '}
        <code>setBytes</code>. The struct is 904 bytes, under <code>setBytes</code>&rsquo;s limit.
      </p>
      <Today checked="2026-11-20 (ch07-facts A4, B2, C1)">
        <p>
          <code>setBytes</code> takes under 4 KB. AMD documents no argument limit and no <code>__grid_constant__</code>,
          so the struct is a plain <code>const</code> parameter.
        </p>
      </Today>
      <p>
        Run <code>python3 labs/pmpp.py run 7.1 --backend emu</code> first, then without <code>--backend</code>. For each
        kernel the tests take the book&rsquo;s 7×7 image and 5×5 filter and check <code>P[2][2] = 321</code> and{' '}
        <code>P[1][0] = 112</code>, then random images with random filters at radii 0, 1, 2, 3 and 7, on shapes from 1×1
        to 257×129 that no block size divides. A random filter is not symmetric, so a flipped filter or one read as{' '}
        <code>F[fCol][fRow]</code> fails there although it passes the book&rsquo;s example. <code>P</code> starts as NaN,
        so an output you never write is caught. On the emulator, a missing ghost test or output guard stops the run with
        an AddressSanitizer report that names the line.
      </p>
      <p>
        With <code>--bench</code>, each radius in 1, 2, 3, 5 and 7 is timed on an image that grows until the basic kernel
        takes about 50 ms. GFLOP/s counts every tap, ghost taps included, as the book does on p. 167. The last column of
        the table divides each kernel&rsquo;s GFLOP/s by the ceiling its model gives it on the same device: its FLOP/B
        times the bandwidth lab 5.1 measured there. Above 1, something served the reuse that the model charges to DRAM.
      </p>
      <LabPrediction
        id="ch07/labs/p-lab1-const-speedup"
        prompt={
          <>
            Before you run the benchmark. At r = 2 (a 5×5 filter), the constant-memory kernel will run how many times as
            fast as the basic kernel on your device? The model&rsquo;s answer is 2, if both kernels are limited by memory
            and nothing else is cached.
          </>
        }
        unit="× the basic kernel"
        measure={constSpeedup}
        reading={
          <>
            The model&rsquo;s 2 counts what each kernel asks of memory. If the basic kernel&rsquo;s filter reads were
            already being answered by a cache, constant memory has little left to save, and the factor shrinks towards
            1. Look at the other radii in the table before you explain the number.
          </>
        }
      />
      <LabPrediction
        id="ch07/labs/p-lab1-byvalue-vs-const"
        prompt={
          <>
            Before you run the benchmark. At r = 2, the by-value kernel&rsquo;s GFLOP/s divided by the constant-memory
            kernel&rsquo;s will be what? On CUDA both read through constant memory. On Metal one is a{' '}
            <code>constant</code> buffer and the other a fixed-size struct filled with <code>setBytes</code>; Apple
            documents no constant cache.
          </>
        }
        unit="× the constant-memory kernel"
        measure={byValueOverConst}
        reading={
          <>
            1 means no difference. Apple&rsquo;s only words on this are a 2016 talk: data in the <code>constant</code>{' '}
            address space whose reads are statically bounded, such as a fixed-size struct passed by reference, can be
            preloaded into registers. Whether your Mac does that is not documented, and one run is not enough to tell a
            few percent from noise: run the benchmark twice.
          </>
        }
      />
      <Lab71Results />
      <p>
        On CUDA there is one more place to look. <code>nvcc -Xptxas=-v</code> lists each kernel&rsquo;s register, shared
        and constant memory, and the README shows where <code>F_c</code>&rsquo;s 900 bytes and the struct&rsquo;s 904
        should appear. That command has not been run for this course, which had no CUDA toolkit.
      </p>
      <p>
        And one question to take away, from the README&rsquo;s last section: llama.cpp keeps its quantisation codebooks
        in global memory on CUDA and HIP, not in <code>__constant__</code>. Each thread looks up its own code.{' '}
        <a href={href('ch07', '7.4')}>Section 7.4</a> explains why that is the constant cache&rsquo;s worst case.
      </p>
      <Beyond>
        <p>
          llama.cpp&rsquo;s tables are <code>static const __device__</code> arrays on CUDA and HIP and program-scope{' '}
          <code>constant</code> arrays on Metal (ch07-facts D4, checked 2026-11-20).
        </p>
      </Beyond>
      <ReferenceSolution lab={l1} tiers={TIERS_71} />

      {/* ------------------------------------------------------------ 7.2 */}
      <h2>Lab 7.2: tiled convolution with halos</h2>
      <p>
        Two tiled kernels, both reading the filter from constant memory. In part A you write the halo kernel of Fig.
        7.12: the block is the input tile, every thread loads one element and stores a zero for a ghost cell, one
        barrier, and then the outer ring of threads sits idle while the others compute. In part B you repair the
        cached-halo kernel of Fig. 7.15, which compiles and has two planted defects. One of them is printed in the book.
      </p>
      <Source section="7.5–7.6" pages="172–179" figs="Figs. 7.12 and 7.15" />
      <p>
        The radius is a template parameter, <code>template &lt;int FILTER_RADIUS&gt;</code>, so the filter loops have
        fixed trip counts, and a provided switch picks the compiled kernel for the radius asked for, as the real kernel
        in <a href={href('ch07', '7.5')}>section 7.5</a>&rsquo;s LLM-serving example does. The tile width is the build&rsquo;s{' '}
        <code>TILE_WIDTH</code>: 32 by default, with 8 and 16 also built for the benchmark. On Metal it is a
        preprocessor macro, because a function constant cannot size a <code>threadgroup</code> array.
      </p>
      <Today checked="2026-11-20 (ch07-facts B6, C3)">
        <p>
          HIP refuses a block larger than the kernel&rsquo;s launch bound, the device maximum by default.
        </p>
      </Today>
      <LabFiles
        lab={l2}
        files={[
          [
            'kernel.cu',
            <>
              part A: <code>convTiledHaloKernel</code> and its grid and block in <code>launchHalo</code>; part B:{' '}
              <code>convCachedKernel</code>, to repair. Leave the code after <code>// @part provided</code> alone.
            </>,
          ],
          ['kernel.metal', <><code>conv_tiled_halo</code> to write, <code>conv_tiled_cached</code> with the same two defects</>],
          ['launch.mm', <>the grid and threadgroup sizes in <code>encodeConvTiledHalo</code></>],
        ]}
      />
      <p>
        Run <code>python3 labs/pmpp.py run 7.2</code>, starting with <code>--backend emu</code>. The tests run both
        kernels at every radius from 1 to the largest that leaves an output tile of at least 2: the book&rsquo;s 7×7
        example (321 and 112 again), then random data with random filters on shapes up to 1000×999, chosen so that tiles
        overhang the right and bottom edges. Part A&rsquo;s tests come first and fail until you have written it; that
        does not stop the run. A tile is tested only where the device takes a block that size: on Metal where the
        pipeline&rsquo;s <code>maxTotalThreadsPerThreadgroup</code> allows it, and on HIP, which may refuse a 32×32
        block, the driver says so.
      </p>
      <p>Part A has three things to get right, and the emulator catches each:</p>
      <ul>
        <li>
          Copy <code>threadIdx</code> into <code>int</code>s before subtracting the radius. The built-ins are unsigned, and
          so are Metal&rsquo;s position attributes: for thread 0, <code>threadIdx.x - FILTER_RADIUS</code> is about four
          billion, not &minus;2.
        </li>
        <li>
          Every thread reaches the barrier. The ring threads compute nothing, but they load halo elements that the others
          read, so the compute guard goes after <code>__syncthreads()</code>.
        </li>
        <li>
          Store the ghost zeros. The compute loop has no bounds test. The emulator gives every block fresh shared
          memory filled with <code>0xCD</code> bytes, so a missing zero fails there every time; on a GPU, memory nobody
          wrote may happen to read as zero, and the mistake can pass.
        </li>
      </ul>
      <p>For part B, the method of lab 5.4:</p>
      <ol>
        <li>
          Run on the emulator first. AddressSanitizer stops the run at the first access outside an array and names the
          line; the tests after that do not run.
        </li>
        <li>Fix one thing, run again, and note how the failures change.</li>
        <li>
          Then run on your GPU, and note which defect it showed you and how. One defect gives wrong outputs almost
          everywhere; the other only near the right edge, in tiles that overhang it.
        </li>
      </ol>
      <p>
        A third thing to notice, which no test can catch: the starter&rsquo;s in-tile test is the book&rsquo;s,{' '}
        <code>threadIdx.x - FILTER_RADIUS + fCol &gt;= 0</code>. Its left side is unsigned, so the comparison is always
        true, and the kernel is correct only because the <code>&lt; TILE_DIM</code> half rejects the wrapped values.
        Fix it anyway (<a href={href('ch07', '7.6')}>section 7.6</a> shows why).
      </p>
      <p>
        With <code>--bench</code>, each tile width and each radius in 1, 2, 3, 5 and 7 that leaves an output tile of at
        least 2 is timed against a provided constant-memory kernel with the same compile-time radius. Each kernel is
        checked on a 100×61 image first, and a wrong one is reported, not timed. Beside every halo row the driver records
        the section 7.5 formula&rsquo;s FLOP/B and its ratio to the constant-memory kernel&rsquo;s 0.5: the factor of
        requested traffic the tiling removes.
      </p>
      <LabPrediction
        id="ch07/labs/p-lab2-halo-speedup"
        prompt={
          <>
            Before you run the benchmark. With a 32×32 input tile and a 5×5 filter, the formula says the halo kernel asks
            for 5.42 / 0.5, about 10.8 times, less global traffic than the constant-memory kernel. How many times as fast
            will it run on your device?
          </>
        }
        unit="× the constant-memory kernel"
        measure={haloSpeedup}
        reading={
          <>
            The 10.8 is about requested bytes. Caches already serve many of the constant-memory kernel&rsquo;s neighbour
            reads (p. 168), and a kernel that asks for little traffic is then limited by something else, such as the
            barrier and the idle ring. Section 7.5 plots every tile and radius against the formula.
          </>
        }
      />
      <LabPrediction
        id="ch07/labs/p-lab2-cached-vs-halo"
        prompt={
          <>
            Before you run the benchmark. At tile 32 and r = 2, the cached-halo kernel&rsquo;s GFLOP/s divided by the
            halo kernel&rsquo;s will be what? Above 1 means the cache served the halo well enough to beat staging it.
          </>
        }
        unit="× the halo kernel"
        measure={cachedOverHalo}
        reading={
          <>
            Section 7.6&rsquo;s kernel bets that a block&rsquo;s halo is still in a cache, and pays for a three-way branch
            inside the filter loop. Timing alone does not say which effect won; a profiler&rsquo;s L2 hit rate helps.
            Section 7.6 shows the ratio for every tile and radius you measured.
          </>
        }
      />
      <LabResults
        lab={l2}
        cols={[
          { head: 'Variant', cell: (r) => KERNEL_72[String(r.name)] ?? String(r.name) },
          { head: 'Tile', num: true, cell: (r) => (r.name === 'const' ? '' : blank(r.tile)) },
          { head: 'Output tile', num: true, cell: (r) => (r.name === 'const' ? '' : blank(r.out_tile)) },
          rawCol('r', 'radius'),
          image,
          msCol,
          numCol('GFLOP/s', 'gflops'),
          numCol('× const', 'speedup_vs_const'),
          numCol('Model FLOP/B', 'model_intensity'),
          numCol('Traffic ratio', 'model_traffic_ratio'),
        ]}
      />
      <p className="muted">
        The provided constant-memory kernel uses 16×16 blocks in every build, so its rows have no tile. On CUDA and HIP
        each timed run also includes the copy of the filter into constant memory, for all three kernels alike.
      </p>
      <ReferenceSolution lab={l2} tiers={TIERS_72} />

      <h2>No project milestone in this chapter</h2>
      <p>
        <code>PLAN.md</code> gives chapter 7 no milestone of the through-line project. Both labs still follow its
        conventions, so later chapters can reuse them: row-major <code>float</code> tensors, raw device pointers and
        explicit sizes as arguments, a <code>launch</code> function beside each kernel, and the CPU reference in{' '}
        <code>main.cpp</code>. The <a href={href('ch07', 'wrap')}>summary</a> lists what of this chapter you will use
        again.
      </p>
    </LabsPage>
  );
}
