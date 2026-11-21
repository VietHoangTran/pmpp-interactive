import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Beyond, Source } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { hOut, lenetLayer, wOut } from '../../sim/cnnlayer';
import { geometry } from '../../sim/convgrid';
import { BOOK_SHAPE, gemmShape, xToB } from '../../sim/unfold';

// A kernel of our own, for one question: the backward pass of Exercise 3 written as a
// scatter. Checked copy: labs/ch19/snippets/wrap_backward_scatter.cu (Wrap.snippets.test.ts
// ties the two together and pins the numbers below to the models).
const SCATTER = `// For a question on chapter 19's summary page (Exercise 3, the backward pass); not built
// or run. The kernel is ours, not the book's. It computes dE/dX, the gradient with respect
// to the layer's input, by giving each thread one element of dE/dY and scattering that
// element's contributions into dE/dX, which is zeroed before the launch. Grid: x covers
// the H_out * W_out pixels of one output map, y the M output maps, z the N samples.
// Layouts as in lab 19.1: NCHW, row-major float. The question asks what is wrong with it.
__global__ void convBackwardXScatter(const float* dY, const float* F, float* dX,
                                     int C, int H, int W, int M, int K) {
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y;
    int n = blockIdx.z;
    if (pixel >= H_out * W_out) return;
    int h = pixel / W_out;
    int w = pixel % W_out;
    float g = dY[((n * M + m) * H_out + h) * W_out + w];
    for (int c = 0; c < C; ++c)
        for (int p = 0; p < K; ++p)
            for (int q = 0; q < K; ++q)
                dX[((n * C + c) * H + h + p) * W + w + q] += g * F[((m * C + c) * K + p) * K + q];
}`;

// ---------------------------------------------------------------- the numbers the questions take from the models
const C5 = lenetLayer('C5')!;
const c5Launch = (division: 'floor' | 'ceil') =>
  geometry({ N: 1, M: C5.M, Hout: hOut(C5), Wout: wOut(C5), tile: 16, division });
/** LeNet C5 on the direct kernel at batch 1, TILE_WIDTH 16 (src/sim/convgrid.ts). */
export const C5_DIRECT = {
  outputs: C5.M * hOut(C5) * wOut(C5),
  threads: c5Launch('ceil').threads,
  printedBlocks: c5Launch('floor').blocks,
  pct: (100 * C5.M * hOut(C5) * wOut(C5)) / c5Launch('ceil').threads,
  inner: C5.C * C5.K * C5.K,
};
/** Threads of the scatter kernel that add into one dX element, book example (src/sim/unfold.ts). */
export const SCATTER_WRITERS = {
  centre: BOOK_SHAPE.M * xToB(BOOK_SHAPE, 0, 1, 1).length,
  corner: BOOK_SHAPE.M * xToB(BOOK_SHAPE, 0, 0, 0).length,
};
/** A depthwise layer, 256 channels of 14 × 14 with 3 × 3 filters, per channel; and the dense layer of the same size. */
export const DEPTHWISE = {
  per: gemmShape({ N: 1, C: 1, H: 14, W: 14, M: 1, K: 3 }),
  dense: gemmShape({ N: 1, C: 256, H: 14, W: 14, M: 256, K: 3 }),
};
/** Fig. 19.5 as printed on LeNet C1's 28 × 28 output maps (src/sim/convgrid.ts). */
const C1 = lenetLayer('C1')!;
const c1Printed = geometry({ N: 1, M: 1, Hout: hOut(C1), Wout: wOut(C1), tile: 16, division: 'floor' });
export const C1_PRINTED = {
  pixels: hOut(C1) * wOut(C1),
  missed: hOut(C1) * wOut(C1) - c1Printed.threads,
};
const r2 = (v: number) => Math.round(v * 100) / 100;
const shape = (g: { rows: number; inner: number; cols: number }) => `${g.rows} × ${g.inner} × ${g.cols}`;

export const questionsWrap: Record<string, Question> = {
  'ch19/wrap/q-c5-direct': {
    kind: 'numeric',
    id: 'ch19/wrap/q-c5-direct',
    prompt: (
      <>
        LeNet&rsquo;s C5 turns 16 maps of 5 × 5 into {C5.M} maps of 1 × 1 with 5 × 5 filters. You run it at batch 1 on
        lab 19.1&rsquo;s direct kernel: <code>TILE_WIDTH</code> 16, the tile count rounded up, every output
        bounds-checked. What percentage of the launched threads compute an output?
      </>
    ),
    answer: r2(C5_DIRECT.pct),
    tol: 0.02,
    unit: '%',
    explain: (
      <p>
        Each output map is one pixel, so it needs one tile: the grid is ({C5.M}, 1, 1) blocks of 256 threads,{' '}
        {C5_DIRECT.threads.toLocaleString('en-US')} threads for {C5_DIRECT.outputs} outputs, {r2(C5_DIRECT.pct)}%. Fig.
        19.5 as printed is worse: 1 / 16 truncates to 0 tiles, so it launches {C5_DIRECT.printedBlocks} blocks and computes
        nothing. C5 is a linear layer in disguise. Each output is a dot product of all {C5_DIRECT.inner} inputs of the
        sample with one row of F, and F is already stored ({C5.M}, {C5_DIRECT.inner}). So{' '}
        <code>
          launchLinear(X, F, Y, N, {C5_DIRECT.inner}, {C5.M})
        </code>{' '}
        computes the layer for the whole batch with no unroll at all, and puts the batch where the GEMM can reuse each
        weight. Section <a href={href('ch19', '19.2')}>19.2</a> has the grid; the C5 reading is ours.
      </p>
    ),
    source: 'Figs. 19.5 and 19.7, pp. 461–462; LeNet C5, p. 458; the linear-layer reading is ours',
  },
  'ch19/wrap/q-subsample': {
    kind: 'choice',
    id: 'ch19/wrap/q-subsample',
    prompt: (
      <>
        Exercise 1 asks for LeNet&rsquo;s subsampling layer: each output pixel is the mean of a 2 × 2 window of one
        input map, the windows do not overlap, and a per-map bias and a sigmoid follow. You give each thread one output
        pixel, as the direct kernel does. What would staging input tiles in shared memory gain?
      </>
    ),
    options: [
      {
        label: 'About a factor of 4 in global loads, since each output reads 4 inputs.',
        why: 'Shared memory saves a load only when a second thread wants the same value. Here no input is in two windows.',
      },
      {
        label: 'Reuse across output maps, which the direct convolution kernel could not get.',
        why: 'Output map m of a subsampling layer reads input map m only. There is nothing to share across maps.',
      },
      {
        label: 'Nothing: each input value is read by one thread, once, which is already the minimum.',
        correct: true,
      },
      {
        label: 'Correctness: without it the 4 threads that add into one output would race.',
        why: 'One thread owns each output and sums its 4 inputs in a register. No two threads write the same element.',
      },
    ],
    explain: (
      <p>
        Overlap is what made tiling pay in chapter 7 and what made B larger than X in section{' '}
        <a href={href('ch19', '19.3')}>19.3</a>. A window whose stride equals its size has none: the expansion ratio is 1,
        as the stride = K setting of 19.3&rsquo;s explorer shows. Like the matrix addition of chapter 5&rsquo;s first
        exercise, this kernel is memory-bound by nature, and the work is in making its loads coalesce. In a real network you would fuse it
        into the convolution before it, so the convolution&rsquo;s output never makes the round trip to global memory.
      </p>
    ),
    source: 'Exercise 1, p. 474; the subsampling layer, Appendix B pp. 616–617 (Fig. B.7); the fusion remark is ours',
  },
  'ch19/wrap/q-backward': {
    kind: 'choice',
    id: 'ch19/wrap/q-backward',
    prompt: (
      <>
        The kernel above is a first attempt at the input gradient of Exercise 3. <code>dX</code> is zeroed before the
        launch, the indices are right, and it runs without errors. Is it correct?
      </>
    ),
    options: [
      {
        label: 'Yes: each thread reads only its own element of dY, so the threads are independent.',
        why: 'The reads are not the problem. Look at which elements of dX two different threads write.',
      },
      {
        label: 'Yes, as long as each block holds one whole output map, so no two blocks meet.',
        why: 'Threads of one map still collide whenever K > 1, because neighbouring output pixels have overlapping patches, and blocks for different maps m also add into the same dX elements.',
      },
      {
        label: 'No: it needs a __syncthreads() before the loops, so all reads of dY finish first.',
        why: 'The colliding threads are also in different blocks (different m), and a barrier would not make two read-add-writes of one element atomic anyway.',
      },
      {
        label: 'No: overlapping patches, and different output maps, add into one dX element without atomics.',
        correct: true,
      },
    ],
    explain: (
      <p>
        In the book&rsquo;s example the centre pixel of each input map lies in 4 patches of each of the 2 output maps,
        so {SCATTER_WRITERS.centre} threads read, add and write it; a corner pixel has {SCATTER_WRITERS.corner}. It is
        section <a href={href('ch19', '19.2')}>19.2</a>&rsquo;s remark about the c, p and q loops seen from the other
        side. Two fixes: make the update an <code>atomicAdd</code>, or turn the kernel round and give each thread one
        element of dX, gathering over m, p and q the terms <code>dY[n, m, h − p, w − q] · F[m, c, p, q]</code> whose
        output pixel exists. The gather has one writer per element and no atomics, and it is the form Appendix B&rsquo;s
        Eq. B.22 is written in. The emulator&rsquo;s <code>race</code> flavour, which runs threads concurrently
        under ThreadSanitizer, is the tool that would report this kernel; it was not run on it.
      </p>
    ),
    source: 'Exercise 3, p. 475; atomics for the inner loops, p. 460; Eq. B.22, Appendix B p. 618; the kernel is ours',
  },
  'ch19/wrap/q-depthwise': {
    kind: 'choice',
    id: 'ch19/wrap/q-depthwise',
    prompt: (
      <>
        Beyond the book: a depthwise convolution gives each of 256 channels its own 3 × 3 filter, so output map m reads
        input map m only. On 14 × 14 maps with no padding, lowered the chapter&rsquo;s way, it is one small GEMM per
        channel. What shape is each (rows × reduction × columns), and what follows?
      </>
    ),
    options: [
      {
        label: `${shape(DEPTHWISE.dense)}, the same as a dense layer of this size, so it is just as good a fit.`,
        why: 'That is the dense layer, where every output map reads all 256 input maps. Here each reads one.',
      },
      {
        label: `${shape(DEPTHWISE.per)}: a matrix-vector product per channel, so each B element is used once.`,
        correct: true,
      },
      {
        label: `256 × 9 × ${DEPTHWISE.per.cols}: one GEMM for all channels, with one B shared by every filter.`,
        why: 'Each channel unfolds a different input map, so there is no single B for all 256 rows of F.',
      },
      {
        label: `1 × 2304 × ${DEPTHWISE.per.cols}, since C·K·K is still 256 · 9.`,
        why: 'Each output reads one channel, so its dot product has 9 terms, not 2304.',
      },
    ],
    explain: (
      <p>
        One output map per channel gives F one row: {shape(DEPTHWISE.per)}, against {shape(DEPTHWISE.dense)} for the
        dense layer. A tiled GEMM reuses each loaded B element across the rows of its output tile, and here there is
        one row, so 15 of every 16 rows of a 16 × 16 tile hold nothing. This is section{' '}
        <a href={href('ch19', '19.3')}>19.3</a>&rsquo;s p. 472 argument at its extreme: small matrices lose. A direct
        kernel with chapter 7&rsquo;s shared-memory tiles fits a depthwise layer, since its windows overlap and each
        input is used by 9 outputs of its own channel.
      </p>
    ),
    source: 'reuse and matrix size, p. 472; tiling, ch. 7 §7.5; depthwise convolution is ours',
  },
  'ch19/wrap/q-int8-scale': {
    kind: 'choice',
    id: 'ch19/wrap/q-int8-scale',
    prompt: (
      <>
        Beyond the book: you store F as int8 with one float scale per output map, and X stays float. In the corrected
        implicit GEMM of section 19.3, where is the cheapest correct place to apply the scale?
      </>
    ),
    options: [
      {
        label: 'In the X gather: multiply each value of X by the scale as it is loaded.',
        why: 'An X value feeds every output map in the tile, and each map has its own scale. One multiply here would give all of them the same one.',
      },
      {
        label: 'In the F tile load: convert each int8 to float and multiply it by its row’s scale.',
        why: `Correct, but it costs a multiply for every F element a thread loads, once per phase (${DEPTHWISE.dense.phases} phases on a 256-channel 3 × 3 layer), for a factor that is the same along the whole row.`,
      },
      {
        label: 'Once per output: multiply Pvalue by scale[Row] just before the store.',
        correct: true,
      },
      {
        label: 'Inside the inner loop over k, so the products stay in range.',
        why: 'Pvalue is a float accumulator, so range is not the issue, and this would cost 16 multiplies per phase.',
      },
    ],
    explain: (
      <p>
        Row m of Y is the dot product of row m of F with a column of B. A factor that is the same for every element of
        the row comes out of the sum: Σ s·f·b = s·Σ f·b. So the int8 values are converted in the F tile load, the sum
        runs as before, and the scale is applied once per output. If the scale changed along the reduction, one per
        group of input channels say, it would no longer factor out, and the tile-load slot is where it would go. That
        slot, where Fig. 19.11 does its gather, is where a quantised GEMM does its dequantising.
      </p>
    ),
    source: 'Fig. 19.11, p. 471; quantisation is ours (notes/ch19.md, relevance section)',
  },
};

// ---------------------------------------------------------------- the page
export function Wrap() {
  const { progress } = useProgress();
  const { reports, loading } = useReports();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const passing = (id: string) => backends.filter((b) => labStatus(reports[b], id, 'work') === 'pass');
  const lab = passing('ch19/lab1_convgemm');
  const m2 = passing('ch06/lab4_linear');
  const name = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch19/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  const q = questionsWrap;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <p>
        The book&rsquo;s summary recaps the layer, the direct kernel, the GEMM form with its input unrolled implicitly,
        and cuDNN, and notes that framework users get all of it without writing a kernel. The parts to keep:
      </p>
      <Source section="19.5" pages="474" />
      <ul>
        <li>
          A convolutional layer is <code>Y[n, m, h, w] = Σ X[n, c, h + p, w + q] · F[m, c, p, q]</code> over c, p and q:
          no flip, no padding, an output K − 1 rows and columns smaller than its input. X and Y carry the batch; F does not.
        </li>
        <li>
          Four of its seven loops, n, m, h and w, write different outputs and are free to parallelise. The other three
          add into one output: keep them serial in a register, or pay for atomics.
        </li>
        <li>
          The direct kernel gives each thread one output on a grid of (output map, tile, sample). Round the tile count
          up and check every output. It requests 0.25 FLOP/B, and it cannot share an input tile across output maps
          because each block owns one map.
        </li>
        <li>
          F is already an <code>M × C·K·K</code> matrix with no copy. Unfold the input into B,{' '}
          <code>C·K·K × H_out·W_out</code>, and the layer is <code>Y = F · B</code>, already in NCHW. Stored, B costs up
          to K² times the input and an extra pass; gathered tile by tile with Eq. 19.5, it costs integer arithmetic in
          the tile load and nothing else.
        </li>
        <li>
          The tile load of a tiled GEMM is a slot where an operand can be transformed without touching the inner loop:
          a gather here, a dequantisation in a quantised kernel.
        </li>
        <li>
          The GEMM form pays when the matrices are large. Small layers, one-row GEMMs and per-sample launches favour a
          direct kernel, and libraries choose per shape: cuDNN and MIOpen among their algorithms, llama.cpp by a rule on
          C·K·K and the output size.
        </li>
        <li>
          Read a library&rsquo;s names before its numbers: cuDNN&rsquo;s K is the chapter&rsquo;s M, and its R and S are
          the chapter&rsquo;s K. Its output size has a formula, and its choice of algorithm can include the precision.
        </li>
      </ul>

      <h2>A mixed check</h2>
      <p>
        These use the whole chapter on layers and kernels the sections did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
      </p>
      <Beyond>
        <p>The depthwise layer, the int8 filters and the scatter kernel are ours, picked for the questions.</p>
      </Beyond>
      <Ask q={q['ch19/wrap/q-c5-direct']} />
      <Ask q={q['ch19/wrap/q-subsample']} label="Exercise 1, as a design question" />
      <Code
        tabs={[
          {
            label: 'CUDA (not a lab file: for the next question only)',
            code: SCATTER,
            status: 'front-end-checked; never run',
            tiers: { dY: 'global', F: 'global', dX: 'global', g: 'reg', h: 'reg', w: 'reg', pixel: 'reg' },
          },
        ]}
        caption={
          <>
            From <code>labs/ch19/snippets/wrap_backward_scatter.cu</code>.
          </>
        }
      />
      <Ask q={q['ch19/wrap/q-backward']} label="Exercise 3, as a design question" />
      <Ask q={q['ch19/wrap/q-depthwise']} />
      <Ask q={q['ch19/wrap/q-int8-scale']} />

      <h2>The book&rsquo;s exercises</h2>
      <p>
        The chapter ends with four exercises (pp. 474–475), and the book gives no solution for any of them. Two are
        worked in the sections; the two that ask for whole kernels, the subsampling layer and the backward pass, are
        worked here as design questions and not built as labs.
      </p>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }} data-table="exercises">
          <thead>
            <tr>
              <th>Exercise</th>
              <th>About</th>
              <th>Where it is worked here</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>1</td>
              <td>The forward pass of the subsampling layer</td>
              <td>
                the subsampling question above; the sequential reference is Appendix B&rsquo;s Fig. B.7 (p. 617), and the
                grid is the one of <a href={href('ch19', '19.2')}>19.2</a> on the smaller map
              </td>
            </tr>
            <tr>
              <td>2</td>
              <td>NHWC and CHWN layouts instead of NCHW</td>
              <td>
                the warp-reads explorer and the layout question in <a href={href('ch19', '19.2')}>19.2</a>, including
                the mode with neighbouring threads on different samples that CHWN needs
              </td>
            </tr>
            <tr>
              <td>3</td>
              <td>The backward pass of the convolutional layer</td>
              <td>
                the scatter-kernel question above, and its gather fix; Appendix B&rsquo;s Figs. B.10 and B.11 (pp.
                619–620) are the sequential references
              </td>
            </tr>
            <tr>
              <td>4</td>
              <td>Coalescing and bank conflicts in Fig. 19.11</td>
              <td>
                the warp explorer and the Exercise 4 question in <a href={href('ch19', '19.3')}>19.3</a>, on NVIDIA&rsquo;s
                documented 32 banks
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Where the PDF looks wrong</h2>
      <p>
        The copy this course was built from has a few apparent slips in chapter 19. Three are in the code and matter:
        the sections and lab 19.1 use corrected versions and say where they differ. The rest are listed so that you are
        not thrown by them when you read the book alongside.
      </p>
      <Aside>
        <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
      </Aside>
      <ul data-list="errata">
        <li>
          Fig. 19.5 (p. 461) divides the output size by <code>TILE_WIDTH</code> with truncation, so LeNet C1&rsquo;s 28 ×
          28 output gets one tile per side and {C1_PRINTED.missed} of its {C1_PRINTED.pixels} pixels are never computed. Section{' '}
          <a href={href('ch19', '19.2')}>19.2</a> and the lab round up.
        </li>
        <li>
          Fig. 19.7 (p. 462) has no bounds check, and its signature lacks the H, W and M that its indices need once they
          are linearised. Fig. 19.5 (p. 461) also leaves <code>W_grid</code>, <code>H_grid</code> and <code>T</code> untyped, and
          Fig. 19.6 (p. 462) labels T as <code>W_grid * H_grid</code> where the code has <code>H_grid * W_grid</code> (the same
          value).
        </li>
        <li>
          Fig. 19.11 (p. 471) truncates the phase count, so C·K·K = 25 runs one phase and drops 9 terms of every output;
          it has no bounds checks and does not show its launch. Section <a href={href('ch19', '19.3')}>19.3</a>&rsquo;s
          erratum hunt shows each failure; the lab provides a corrected, tested version.
        </li>
        <li>
          p. 468 lists the output tile of Fig. 19.9 as four pixels of output map 0. The red box and the B tile it uses
          give <code>Y[n, 0, 0, 0]</code>, <code>Y[n, 0, 0, 1]</code>, <code>Y[n, 1, 0, 0]</code> and{' '}
          <code>Y[n, 1, 0, 1]</code>: the values 14, 20 / 12, 24.
        </li>
        <li>
          p. 469 says &ldquo;u is equal to 2&rdquo; for <code>B[10, 2]</code>. It is u = 10 and v = 2; the 2 that the
          arithmetic produces is the channel c. The same page writes <code>B[3.0]</code> for <code>B[3, 0]</code>.
        </li>
        <li>
          Eq. 19.1 writes each term as patch · filter and Eq. 19.2 as filter · patch, and the sentence after Eq. 19.3
          names its operands in the opposite order to the equation. The value, 14, is the same either way. On p. 464,
          &ldquo;the expression for generating the Y0,0,0&rdquo; drops the n subscript used a line earlier, and
          &ldquo;the layout of C matrices&rdquo; means arrays in the C language, not a matrix named C.
        </li>
        <li>
          p. 458 says the filter bank is the array parameter <code>W</code>; Fig. 19.3 calls it F, and W is the width. It
          also says a placeholder index runs &ldquo;from 0 to K&rdquo;: filter indices stop at K − 1.
        </li>
        <li>
          p. 458 calls F6&rsquo;s weight matrix 120 × 84, where <code>W · X</code> with 120 inputs and 84 outputs needs
          84 × 120 in the usual order. Fig. 19.1 labels C5 &ldquo;Full connection&rdquo; while the text calls it a
          convolutional layer; both are right for a 5 × 5 filter on a 5 × 5 map.
        </li>
        <li>
          p. 458 leaves the fully connected layer &ldquo;as an exercise&rdquo;, and Exercise 1 says the subsampling
          layer is introduced in section 19.1. No exercise asks for the first, and Appendix B describes the second.
        </li>
        <li>
          p. 454 promises &ldquo;a few optimizations&rdquo; of the basic kernel. The only one developed is the GEMM form;
          tiling the direct kernel is left to you (p. 463).
        </li>
        <li>
          p. 472 says cuDNN needs its data in device memory &ldquo;as we discussed earlier in this chapter&rdquo;; we found
          no such discussion. p. 473 gives the output size only as <code>P = f(H; R; u; pad_h)</code>; section{' '}
          <a href={href('ch19', '19.4')}>19.4</a> gives cuDNN&rsquo;s formula. The same page gives padding&rsquo;s purpose
          as memory alignment or vectorised execution, where cuDNN documents <code>pad_h</code> and <code>pad_w</code> as
          zero rows and columns around the input (19.4 again), and groups Winograd under the matrix-multiplication
          algorithms.
        </li>
        <li>
          p. 460 opens by comparing &ldquo;training&rdquo; a network with matrix multiplication, before a section about
          the forward path. The text writes CUDNN where its reference writes cuDNN, and section 19.3&rsquo;s
          &ldquo;unfolding&rdquo; becomes &ldquo;unrolling&rdquo; in the summary.
        </li>
      </ul>

      <h2>What this course shortened</h2>
      <p>
        Every section of the chapter has its own page here. Kept to a sentence or a pointer: the history of CNNs (pp.
        454–455); LeNet-5&rsquo;s subsampling, fully connected and Gaussian layers, named but not computed (pp. 455,
        458–459, with Appendix B, which this course skips); the backward path (p. 455), which appears only as the
        Exercise 3 question above; and FFT and Winograd convolution, named in 19.4&rsquo;s table (p. 473). Explained but
        not built: shared-memory tiling of the direct kernel (p. 463), which chapter 7&rsquo;s section{' '}
        <a href={href('ch07', '7.5')}>7.5</a> and lab 7.2 teach; the matmul inside cuDNN (pp. 473–474), which chapter 15
        builds; and strided, padded, dilated and grouped convolutions, which appear in 19.4&rsquo;s calculator and in
        notes beside the explorers but in no lab code. The implicit GEMM of Fig. 19.11 is provided in the lab, corrected
        and tested, rather than written by you.
      </p>

      <h2>The project so far</h2>
      <p>
        This chapter adds no milestone. Lab 19.1 calls milestone M2, lab 6.4&rsquo;s{' '}
        <code>launchLinear(X, W, Y, batch, in, out)</code>, once per sample with F in the place of X and the unrolled
        input in the place of W, and nothing in the GEMM had to change. That is what the project&rsquo;s conventions
        were for: row-major <code>float</code>, raw device pointers, explicit sizes, and weights stored (out, in).{' '}
        <span data-lab19={loading ? 'loading' : lab.join(',') || 'none'}>
          {loading
            ? 'Reading reports.'
            : backends.length === 0
              ? 'No lab results are recorded on this machine yet.'
              : lab.length
                ? <>Your own lab 19.1 kernels pass on {lab.map(name).join(' and ')} on this machine.</>
                : <>Your own lab 19.1 kernels have not passed on this machine yet.</>}
        </span>{' '}
        <span data-m2={loading ? 'loading' : m2.join(',') || 'none'}>
          {loading || backends.length === 0
            ? ''
            : m2.length
              ? <>Your own M2 passes on {m2.map(name).join(' and ')}, so <code>PMPP_CH19_GEMM=work</code> runs the layer on your GEMM.</>
              : <>Your own M2 has not passed yet; until it does, the lab calls lab 6.4&rsquo;s reference.</>}
        </span>{' '}
        See the <a href={href('ch19', 'labs')}>labs</a>.
      </p>
      <p>
        <a href={href('ch20')}>Chapter 20</a> is the model this course is about. Its linear layers are M2 and M3 as they
        stand, its attention scores are matrix products, and its decode steps make the thin, batch-1 shapes that this
        chapter&rsquo;s p. 472 argument warns about. Milestones M6 and M7 begin there.
      </p>
    </>
  );
}
