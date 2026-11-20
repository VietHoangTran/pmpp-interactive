import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { span } from '../../sim/blur';
import { layout, strides } from '../../sim/mapping';

// A kernel of our own, for one question: it normalises each row of a matrix in place.
// Checked copy: labs/ch03/snippets/scale_rows_defect.cu (Wrap.snippets.test.ts ties the two).
const SCALE_ROWS = `// Divide every row of a rows x cols matrix by that row's first element, in place.
__global__ void scaleRowsKernel(float* X, int rows, int cols) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        X[row*cols + col] = X[row*cols + col] / X[row*cols];
    }
}

void launchScaleRows(float* X, int rows, int cols) {
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((cols + 15)/16, (rows + 15)/16, 1);
    scaleRowsKernel<<<dimGrid, dimBlock>>>(X, rows, cols);
}`;

// The numbers the questions ask for, from the models the sections run on.
/** One token through the 1024 to 4096 layer under lab 3.3's launch: x covers l, y covers j, 16 by 16 blocks. */
const DECODE = layout({ height: 1, width: 4096, blockW: 16, blockH: 16, style: 'blocks' });
/** Valid positions along one axis for an element at index 0 with radius 2; a corner of a volume has three such axes. */
const AXIS_AT_CORNER = span(0, 64, 2);
const AXIS_INSIDE = span(32, 64, 2);
const CORNER_3D = AXIS_AT_CORNER ** 3;
/** Strides of a row-major [batch, j, k] tensor, in units of j and k: (j*k, k, 1). */
const BATCH_STRIDES = strides([8, 3, 5]); // batch 8, j = 3, k = 5

const n = (v: number) => v.toLocaleString('en-US');
// Keeps the spaces of an index expression but lets it wrap at one of them on a phone.
const pre = { whiteSpace: 'pre-wrap' as const };

export const questionsWrap: Record<string, Question> = {
  'ch03/wrap/q-inplace-rows': {
    kind: 'choice',
    id: 'ch03/wrap/q-inplace-rows',
    prompt: (
      <>
        <code>launchScaleRows</code> above is called on a 4 × 50 matrix in which every element is 2.0. Which elements
        are guaranteed to hold 1.0 afterwards?
      </>
    ),
    options: [
      {
        label: 'All 200. Each element has exactly one writer, so no order of threads can change the result.',
        why: 'One writer per element is half of the rule. The other half is that no thread reads what another thread writes, and here 196 threads read an element that a different thread stores to.',
      },
      { label: 'Only the four in column 0.', correct: true },
      {
        label: 'Columns 0 to 15: those threads share a block with the thread of column 0.',
        why: 'Sharing a block orders nothing. Without a barrier the threads of one block run in no promised order either.',
      },
      {
        label: 'None of them.',
        why: 'The thread of column 0 reads only its own element, and no other thread writes it: 2.0 / 2.0 is 1.0 in any order.',
      },
    ],
    explain: (
      <>
        <p>
          Every thread divides by <code>X[row*cols]</code>, and the thread of column 0 in that row overwrites it with
          1.0. A thread that reads it before that store computes 2.0 / 2.0 = 1.0. One that reads it afterwards computes
          2.0 / 1.0 = 2.0. Nothing orders the two, so each of the other 196 elements can come out either way. The
          mapping, the guard and the flat index are all correct; the defect is that the kernel reads what it writes.
        </p>
        <p>
          On the course's CPU emulator, which runs one thread at a time, the three thread orders the labs use (in
          order, reversed, shuffled) leave 4, 64 and 37 of the 200 elements at 1.0 (emulator-checked:{' '}
          <code>python3 labs/ch03/snippets/check.py</code>; no GPU was involved). The fix is the one section 3.3 used
          for blur: write to a second buffer, or pass the divisors in an array of their own that no thread writes.
        </p>
      </>
    ),
    source: 'our kernel; the discipline is that of ch. 3 §3.4, p. 60 (one thread, one output element), and of "Why two buffers" in section 3.3',
  },
  'ch03/wrap/q-batched': {
    kind: 'choice',
    id: 'ch03/wrap/q-batched',
    prompt: (
      <>
        A linear layer is applied to a whole batch of sequences. <code>A</code> is a row-major{' '}
        <code>[batch, j, k]</code> tensor, the weight <code>B</code> is one <code>[k, l]</code> matrix shared by every
        entry, and <code>C</code> is <code>[batch, j, l]</code>. A 3D grid gives each thread <code>col</code> from x,{' '}
        <code>row</code> from y and <code>b</code> from z. Which product does the thread add at step <code>i</code> of
        its loop?
      </>
    ),
    options: [
      { label: <code style={pre}>A[(b*j + row)*k + i] * B[i*l + col]</code>, correct: true },
      {
        label: <code style={pre}>A[b*j + row*k + i] * B[i*l + col]</code>,
        why: 'The batch index has to skip whole matrices of j*k elements. Here it skips j elements, so entry 1 starts inside entry 0.',
      },
      {
        label: <code style={pre}>A[(b*k + row)*j + i] * B[i*l + col]</code>,
        why: 'The sizes are in the wrong places: a row of A holds k elements, so k is the stride of the row index.',
      },
      {
        label: <code style={pre}>A[(b*j + row)*k + i] * B[(b*k + i)*l + col]</code>,
        why: 'B has no batch dimension. For any b above 0 this reads past the end of the weight.',
      },
    ],
    explain: (
      <p>
        The stride of an index is the product of the sizes to its right. For <code>[batch, j, k]</code> that is (j·k,
        k, 1): with j = 3 and k = 5 the model gives ({BATCH_STRIDES.join(', ')}). So the element is at{' '}
        <code>b*j*k + row*k + i</code>, which is <code>(b*j + row)*k + i</code>, and that is the book's{' '}
        <code>plane*m*n + row*m + col</code> with other names. The weight is indexed exactly as in lab 3.3, because
        every batch entry multiplies by the same matrix. The store is <code>C[(b*j + row)*l + col]</code>, and the
        guard tests all three of b, row and col.
      </p>
    ),
    source: 'our kernel, built from ch. 3 §3.2, p. 55 (3D form) and §3.4, pp. 60–62 (index reasoning)',
  },
  'ch03/wrap/q-decode-surplus': {
    kind: 'numeric',
    id: 'ch03/wrap/q-decode-surplus',
    prompt: (
      <>
        Lab 3.3 launches 16 × 16 blocks, with the grid's x covering the l outputs and its y covering the j rows of the
        batch. One token goes through a layer of 1024 inputs and 4096 outputs: j = 1, k = 1024, l = 4096. How many of
        the launched threads stop at the guard?
      </>
    ),
    answer: DECODE.surplus,
    unit: 'threads',
    explain: (
      <p>
        The grid is ({DECODE.gridX}, {DECODE.gridY}): {DECODE.blocks} blocks of 256 threads, {n(DECODE.threads)} in
        all. Only the first thread row of each block has an element of C, so {n(DECODE.working)} threads work and{' '}
        {n(DECODE.surplus)} stop at the guard: 15 of every 16. The arithmetic done is the same as with any other
        launch, so this count alone says nothing about time; the batch 1 row of lab 3.3's benchmark is the
        measurement. Blocks of 256 by 1 would cover the same row with no surplus thread at all. Choosing the block
        shape from the shape of the data and from what the device reports is how chapter 4 completes milestone M1.
      </p>
    ),
    source: 'method of ch. 3 §3.2, pp. 50 and 54–55, applied to our lab 3.3',
  },
  'ch03/wrap/q-corner3d': {
    kind: 'numeric',
    id: 'ch03/wrap/q-corner3d',
    prompt: (
      <>
        A kernel averages the (2r + 1)³ values of the cube around each element of a 3D volume and skips the
        positions outside the volume, the way the blur kernel does at the edges of a picture. With r = 2, how many values does the
        thread of a corner element add up?
      </>
    ),
    answer: CORNER_3D,
    unit: 'values',
    explain: (
      <p>
        Take one axis at a time. Around index 0 the loop visits −2, −1, 0, 1, 2 and {AXIS_AT_CORNER} of those exist.
        A corner is at the end of all three axes, and the counts multiply: {AXIS_AT_CORNER}³ = {CORNER_3D} of{' '}
        {AXIS_INSIDE ** 3}. An element on an edge of the volume gets {AXIS_AT_CORNER * AXIS_AT_CORNER * AXIS_INSIDE},
        one in the middle of a face {AXIS_AT_CORNER * AXIS_INSIDE * AXIS_INSIDE}. That is why the inner test has one
        pair of clauses per axis, and why the divisor has to be a count kept by the thread.
      </p>
    ),
    source: 'method of ch. 3 §3.3, pp. 58–59; the 3D kernel is ours (the book returns to 3D arrays in ch. 8)',
  },
  'ch03/wrap/q-squaretest': {
    kind: 'choice',
    id: 'ch03/wrap/q-squaretest',
    prompt: (
      <>
        A kernel for <code>C[j × l] = A[j × k] · B[k × l]</code> reads its second operand as{' '}
        <code>B[i*k + col]</code> where <code>B[i*l + col]</code> is right. Which of these test shapes, written (j, k,
        l), does it pass?
      </>
    ),
    options: [
      {
        label: '(5, 7, 3)',
        why: 'k and l differ, so from i = 1 on each read lands 4·i elements too far along.',
      },
      {
        label: '(1, 40, 33)',
        why: 'One row of C does not help: the stride belongs to B, and 40 is not 33.',
      },
      { label: '(20, 1, 35)', correct: true },
      {
        label: '(33, 20, 1)',
        why: 'B has 20 elements here and the kernel reads B[i*20], which is past its end for every i above 0.',
      },
    ],
    explain: (
      <p>
        The two expressions agree when k = l. They also agree when k = 1, because then i is only ever 0 and the stride
        multiplies nothing. So this kernel passes every square test and the outer product, and fails everything else.
        Lab 3.3's tests include shapes in which all three sizes differ for exactly this reason. When the lab was
        built, this defect was run against them: on Metal it passed the four square shapes and the k = 1 shape, 5 of
        12. It is also what the first test of lab 3.5, the cube, is there to show.
      </p>
    ),
    source: 'our lab 3.3; index reasoning of ch. 3 §3.4, p. 62',
  },
};

const M1 = 'ch03/lab3_naive_gemm';
// Backends that are not a GPU: the emulator and its race-detection flavour.
const EMULATED = new Set(['emu', 'race']);

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const devices = Object.keys(reports).filter((b) => !EMULATED.has(b));
  const m1 = devices.filter((b) => labStatus(reports[b], M1, 'work') === 'pass');
  const m1emu = labStatus(reports.emu, M1, 'work') === 'pass';
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch03/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul>
        <li>
          A launch is two <code>dim3</code> values, a grid of blocks and a block of threads, each up to three
          dimensions. A thread knows only its four built-ins. How large either may be is a property of the device, and
          a launch that breaks a limit fails without a word unless the host asks.
        </li>
        <li>
          <code>row = blockIdx.y*blockDim.y + threadIdx.y</code> and{' '}
          <code>col = blockIdx.x*blockDim.x + threadIdx.x</code>. Rows come from y and columns from x, so the grid's x
          is sized from the width and its y from the height. Labels and data sizes are written highest dimension
          first; the <code>dim3</code> constructor takes x first.
        </li>
        <li>
          A grid is whole blocks, so some threads have no element. Every kernel guards every index it computed. On
          Metal, <code>dispatchThreads</code> makes the grid exactly the data and the guard goes away;{' '}
          <code>dispatchThreadgroups</code> is the book's situation.
        </li>
        <li>
          Memory is flat. An element of a row-major array is at <code>row*width + col</code>, and in three dimensions
          at <code>plane*m*n + row*m + col</code>: the stride of an index is the product of the sizes to its right.
        </li>
        <li>
          A thread may loop. The blur kernel has two tests that prevent two different failures: the outer one keeps
          surplus threads from storing, the inner one keeps a patch from reading outside the picture. Divide by what
          was counted, not by the patch area.
        </li>
        <li>
          Each thread writes its own output element and reads only what no thread writes. Then no barrier, lock or
          atomic is needed and no order of threads changes the result. A kernel that updates its input in place
          breaks the second half of that rule.
        </li>
        <li>
          The naive matmul reads <code>M[row*Width + k]</code> along a row and <code>N[k*Width + col]</code> down a
          column. For rectangular matrices each <code>Width</code> becomes exactly one of j, k and l, and a kernel is
          written for one storage convention: the book's, kept by labs 3.3 and 5.3, holds the weight as (in, out),
          the transpose of what PyTorch and ggml store.
        </li>
        <li>
          A printed listing is not tested code. A cast binds tighter than <code>*</code>, the compiler says nothing,
          and only a test with a known answer shows it. Test with sizes that differ from one another and that are not
          multiples of the block.
        </li>
      </ul>
      <Source section="3.5" pages="64" />

      <h2>A mixed check</h2>
      <p>
        These draw on the whole chapter and on situations the sections did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly on the first recorded try.` : ''}
      </p>
      <Code
        tabs={[
          {
            label: 'CUDA (not a lab file: for this question only)',
            code: SCALE_ROWS,
            tiers: { X: 'global', row: 'reg', col: 'reg' },
            status: 'front-end-checked; has a defect on purpose',
          },
        ]}
        caption={
          <>
            Ours, not the book's. The checked copy is <code>labs/ch03/snippets/scale_rows_defect.cu</code>.
          </>
        }
      />
      <Ask q={questionsWrap['ch03/wrap/q-inplace-rows']} label="Find the defect" />
      <Ask q={questionsWrap['ch03/wrap/q-batched']} />
      <Ask q={questionsWrap['ch03/wrap/q-decode-surplus']} />
      <Ask q={questionsWrap['ch03/wrap/q-corner3d']} />
      <Ask q={questionsWrap['ch03/wrap/q-squaretest']} />

      <h2>The book's exercises</h2>
      <p>
        The chapter ends with five exercises (pp. 64–65), printed straight after the summary with no heading of their
        own. Each is worked somewhere in this course; the book prints no answers, so the answers here are ours.
      </p>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="exercises">
        <thead>
          <tr><th>Exercise</th><th>About</th><th>Where it is worked here</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>1a, 1b</td>
            <td>Matmul with one thread per output row, and per output column</td>
            <td>lab 3.4 on the <a href={href('ch03', 'labs')}>labs page</a>; the granularity explorer in <a href={href('ch03', '3.4')}>3.4</a></td>
          </tr>
          <tr>
            <td>1c</td>
            <td>Pros and cons of the two designs</td>
            <td>a question in <a href={href('ch03', '3.4')}>3.4</a>, then your own measurement in lab 3.4</td>
          </tr>
          <tr>
            <td>2</td>
            <td>Matrix times vector, one thread per output element</td>
            <td>
              a question in <a href={href('ch03', '3.4')}>3.4</a>; the l = 1 test of lab 3.3. The host stub the
              exercise also asks for is chapter 2's allocate, copy, launch, copy back, and is not repeated.
            </td>
          </tr>
          <tr>
            <td>3a, 3b, 3c</td>
            <td>Threads per block, threads and blocks in a grid, read from host code</td>
            <td>the closing question of <a href={href('ch03', '3.1')}>3.1</a></td>
          </tr>
          <tr>
            <td>3d</td>
            <td>How many threads execute the guarded line</td>
            <td>a question in <a href={href('ch03', '3.2')}>3.2</a>, and the "exercise 3" preset of its thread map</td>
          </tr>
          <tr>
            <td>4a, 4b</td>
            <td>Flat index in row-major and in column-major order</td>
            <td>two questions and the flattening explorer in <a href={href('ch03', '3.2')}>3.2</a></td>
          </tr>
          <tr>
            <td>5</td>
            <td>Flat index into a 3D tensor</td>
            <td>a question in <a href={href('ch03', '3.2')}>3.2</a>, stated with strides; the batched question above</td>
          </tr>
        </tbody>
      </table>

      <h2>Where the PDF looks wrong</h2>
      <p>
        This edition is new and the copy this course was built from has apparent slips in chapter 3. The first four are
        in the code of Fig. 3.4 and its call: three change what the program computes and the fourth stops the printed
        call from compiling, which is why the labs do not use the listing as printed. The rest are listed so that you
        are not thrown by them when you read the book alongside.
      </p>
      <ul data-list="errata">
        <li>
          Fig. 3.4, line 19 (p. 53): <code>(unsigned char) 0.21f*r+0.71f*g+0.07f*b</code>. The cast applies to{' '}
          <code>0.21f</code> alone, which becomes 0, so the red term is lost. The whole sum needs parentheses.{' '}
          <a href={href('ch03', '3.2')}>Section 3.2</a> has you find this, and lab 3.1 has the tested version.
        </li>
        <li>
          The same line uses 0.21, 0.71 and 0.07, which add up to 0.99. The text on p. 52 gives 0.299, 0.587 and
          0.114, which is what this course uses.
        </li>
        <li>
          The listing's kernel takes <code>(Pout, Pin, width, height)</code>, and the host call on p. 50 passes{' '}
          <code>(Pin_d, Pout_d, m, n)</code>. As printed, input and output are swapped.
        </li>
        <li>
          The listing names the kernel <code>colortoGrayscaleConvertion</code>; the text, the caption and the host
          call say <code>colorToGrayscaleConversion</code>. The call as printed would not compile against the listing.
        </li>
        <li>
          p. 53 cites "line 07" for the range test, which is on line 08 of Fig. 3.4. The comment on lines 11 and 12
          says CHANNEL where the code has <code>CHANNELS</code>.
        </li>
        <li>
          The caption of Fig. 3.5 says "76 × 62 picture". The text and Fig. 3.2 say 62 × 76, rows first, which is the
          book's own convention (footnote 1, p. 49). The text calls areas 2 and 3 medium shaded and area 4 lightly
          shaded; in our copy areas 2 to 4 look alike, and the borders and numerals are what tell them apart.
        </li>
        <li>p. 48 prints "threadId.x" for <code>threadIdx.x</code>.</li>
        <li>
          Fig. 3.8 (p. 57): the printed line numbers go from 07 to 09, and the comment on line 09 calls the patch a
          "BLUR_SIZE x BLUR_SIZE box". The patch is 2·BLUR_SIZE + 1 on a side, as the text on p. 58 says. The book
          prints no launch for this kernel.
        </li>
        <li>Fig. 3.10 labels the height of a tile "BLOCK_WIDTHE".</li>
        <li>
          Fig. 3.11 names the kernel <code>MatrixMulKernel</code>; the text and the caption of Fig. 3.12 write{' '}
          <code>matrixMulKernel</code>. The chapter never prints the launch or a definition of{' '}
          <code>BLOCK_WIDTH</code> for it; the launch in <a href={href('ch03', '3.4')}>section 3.4</a> is ours.
        </li>
        <li>
          p. 63 computes the row and column of Thread (1,0) of Block (0,0) as "0 ∗ 0 + 1 = 1 and 0 ∗ 0 + 0 = 0". With
          blocks 2 wide the products are 0 ∗ 2; the results are the same. On the same page "k ∗ Width + col2 ∗ 4 + 0 =
          8" is missing an equals sign after "col".
        </li>
        <li>
          Smaller things: the lattice of Fig. 3.9 is not dimensioned, some labels of Fig. 3.13 are hidden under its
          arrows, and a sentence of the BLAS sidebar on p. 60 is missing the word "in".
        </li>
      </ul>
      <Aside>
        <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
      </Aside>
      <p>
        One slip is not the book's. Apple's page on calculating threadgroup and grid sizes prints "a width of 32, a
        height of 48" threadgroups for its 1024 × 768 example, where its own code and values give 64 by 24. Both
        multiply to the 1,536 threadgroups it then states.
      </p>
      <Today checked="2026-11-20 (notes/today/ch03-facts.md, C3)">
        <p>
          The page's values are 512 threads per group and an execution width of 16, so groups 16 wide and 32 high.
          The arithmetic was redone by hand, not run.
        </p>
      </Today>

      <h2>What this chapter left out</h2>
      <ul>
        <li>
          The "Memory Space" sidebar (p. 51) is one sentence in section 3.2, and the "Linear Algebra Functions"
          sidebar (pp. 59–60) one paragraph in section 3.4. Solvers and eigenvalue routines are not discussed.
        </li>
        <li>
          What blurring is used for, and the photograph of Fig. 3.6 (p. 56). The course uses small synthetic pictures
          so that every number on screen can be checked by hand.
        </li>
        <li>Weighted and Gaussian blur (p. 56): chapter 7, with convolution.</li>
        <li>3D arrays in stencil kernels (p. 55): chapter 8.</li>
        <li>
          What the four kinds of block cost when a warp's threads disagree at the guard: chapter 4. What the naive
          matmul's memory traffic costs: chapter 5. How its two access patterns differ in cost: chapter 6.
        </li>
      </ul>

      <h2>The project so far</h2>
      <p>
        Lab 3.3's kernel, a naive GEMM for any shapes with the batch first, is the first half of milestone M1 of the
        through-line project: the computation behind every linear layer of the small transformer you will end up
        running, in its plainest form.{' '}
        <span data-m1>
          {devices.length === 0 && !reports.emu
            ? 'No lab results are recorded on this machine yet.'
            : m1.length
              ? `Your own version passes on ${m1.join(' and ')} on this machine.`
              : m1emu
                ? 'Your own version passes on the emulator; it has not passed on a device on this machine yet.'
                : 'Your own version has not passed on this machine yet.'}
        </span>{' '}
        See the <a href={href('ch03', 'labs')}>labs</a>. The second half is chapter 4's: the 16 × 16 block of this
        chapter is a fixed choice, and there the launch configuration is chosen from what the device reports. Chapter
        5 then replaces the kernel behind the same arguments with a tiled one, milestone M2.
      </p>
      <p>
        Chapter 4 opens the device this chapter treated as a box. It shows how blocks are scheduled, how the threads
        of a block are grouped into warps, and what a branch costs when the threads of one warp take different sides
        of it. This chapter's guard is such a branch, and the four kinds of block from section 3.2 return there as
        the worked example. Lab 3.1's measurement of wide blocks against tall ones is left open until then.
      </p>
    </>
  );
}
