import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { labStatus, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { ceilDiv, launch } from '../../sim/launch';

// Two kernels of our own, each shown only for a question. The checked copies are
// labs/ch02/snippets/saxpy_stub_defect.cu and labs/ch02/snippets/count_above_race.cu
// (Wrap.snippets.test.ts fails if a copy here and its file drift apart).
const SAXPY = `// y = a*x + y for n floats, on the device. It has one defect, on purpose.
// No error checks, to keep the listing short, as in the book's figures.
__global__ void saxpyKernel(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

void saxpy(float a, const float* x_h, float* y_h, int n) {
    float *x_d, *y_d;
    size_t size = (size_t)n * sizeof(float);
    cudaMalloc((void**)&x_d, size);
    cudaMalloc((void**)&y_d, size);
    cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice);

    saxpyKernel<<<(n + 255) / 256, 256>>>(a, x_d, y_d, n);

    cudaMemcpy(y_h, y_d, size, cudaMemcpyDeviceToHost);
    cudaFree(x_d);
    cudaFree(y_d);
}`;

const COUNT = `// Count the probabilities above a threshold t. *count is 0 before the launch.
__global__ void countAbove(const float* p, float t, int* count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && p[i] > t) {
        *count = *count + 1;
    }
}`;

// The numbers the questions ask for.
/** One token's logits through a one-thread-per-element kernel in blocks of 256 (the vocabulary size is our example). */
const VOCAB = 50257;
const LOGITS = launch(VOCAB, 256, ceilDiv(VOCAB, 256));
/** An embedding table of 150,000 tokens by 4,096 dimensions, in float. */
const TABLE = { rows: 150000, cols: 4096 };
const TABLE_N = TABLE.rows * TABLE.cols;
const TABLE_BYTES = TABLE_N * 4;
const INT_MAX = 2 ** 31 - 1;
/** What a 32-bit int holds after the conversion: C++20 defines it modulo 2^32. */
const WRAPPED = TABLE_BYTES > INT_MAX ? TABLE_BYTES - 2 ** 32 : TABLE_BYTES;

const n = (v: number) => v.toLocaleString('en-US');

export const questionsWrap: Record<string, Question> = {
  'ch02/wrap/q-inout': {
    kind: 'choice',
    id: 'ch02/wrap/q-inout',
    prompt: (
      <>
        <code>saxpy</code> above is called on a CUDA GPU with every <code>x_h[i]</code> equal to 1, every{' '}
        <code>y_h[i]</code> equal to 2, and <code>a = 3</code>. What does <code>y_h</code> hold afterwards?
      </>
    ),
    options: [
      {
        label: '5 everywhere. Every call succeeds and every element has its thread.',
        why: 'The launch and the guard are right. But the kernel reads y[i] before it writes it, and nothing put the caller\'s y on the device.',
      },
      { label: '3 plus whatever the allocation of y_d happened to hold, in every element.', correct: true },
      {
        label: 'Exactly 3 everywhere, because cudaMalloc hands out zeroed memory.',
        why: 'NVIDIA documents the opposite: "The memory is not cleared." Zero is what Metal\'s newBufferWithLength gives, and what the emulator\'s fill pattern comes close to.',
      },
      {
        label: 'Nothing: the call fails, because the kernel reads memory that nobody wrote.',
        why: 'No API call reports a read of uninitialised device memory. Every call here returns success.',
      },
    ],
    explain: (
      <>
        <p>
          The stub has the shape of the vector addition stub, where C is only written and so is never copied in (
          <a href={href('ch02', '2.3')}>section 2.3</a>). Here <code>y</code> is read as well as written, so it needs
          the copy in that C did not: <code>cudaMemcpy(y_d, y_h, size, cudaMemcpyHostToDevice)</code> before the
          launch. Without it the kernel adds <code>a * x[i]</code> to whatever <code>y_d</code> held, which CUDA does
          not clear.
        </p>
        <p>
          The dangerous part is that the wrong answer can look right. On the course's emulator, whose fresh device
          memory reads as a float of about −2.9 × 10⁻¹⁶, a one-off run of this file gave exactly 3 in all 1000
          elements (emulator-checked; no GPU was involved). On Metal a buffer made with{' '}
          <code>newBufferWithLength</code> starts at zero and gives the same. A test whose <code>y</code> starts at 0
          passes. The same holds for any operation done in place, lab 2.3's <code>addResidual</code> among them: an
          operand that is updated is an input too.
        </p>
      </>
    ),
    source: 'our kernel; ch. 2 §2.3, p. 27 and §2.4, pp. 30–31; cudaMalloc: notes/today/ch02-facts.md, A2; Metal: C2',
  },
  'ch02/wrap/q-logits': {
    kind: 'numeric',
    id: 'ch02/wrap/q-logits',
    prompt: (
      <>
        A sampler divides each of the {n(VOCAB)} logits of one token by a temperature, one thread per logit, in
        blocks of 256 threads and the fewest blocks that cover them. How many of the launched threads stop at the
        guard?
      </>
    ),
    answer: LOGITS.surplus,
    unit: 'threads',
    explain: (
      <p>
        ({n(VOCAB)} + 255) / 256 = {LOGITS.blocks} blocks in integer division, so {n(LOGITS.threads)} threads.{' '}
        {n(LOGITS.working)} of them have a logit and {LOGITS.surplus} stop at <code>if (i &lt; n)</code>, all in the
        last block. The vocabulary size is our example; any size that is not a multiple of 256 leaves a partly
        filled last block, which is why every kernel of this kind carries the guard. On Metal,{' '}
        <code>dispatchThreads</code> with a grid of exactly {n(VOCAB)} threads launches none spare.
      </p>
    ),
    source: 'method of ch. 2 §2.5, p. 37 and §2.6, p. 38 (the book\'s n = 1000 case); the sampler is ours',
  },
  'ch02/wrap/q-count': {
    kind: 'choice',
    id: 'ch02/wrap/q-count',
    prompt: (
      <>
        <code>countAbove</code> above is launched with 4 blocks of 256 threads over 1000 probabilities, of which 300
        are above <code>t</code>. What can you expect of <code>*count</code> after the kernel has finished?
      </>
    ),
    options: [
      {
        label: 'It is 300: every thread that passes the test adds one.',
        why: 'Each update is a read, an addition and a write. Two threads can read the same old value and both write that value plus one, and one of the additions is lost.',
      },
      {
        label: 'It is 300 if all 300 elements fall in one block, because a block runs its threads in order.',
        why: 'Nothing orders the threads of a block either. A block shares a launch configuration, not a schedule.',
      },
      { label: 'It can be less than 300, and it can change from run to run.', correct: true },
      {
        label: 'It is 0: each thread adds one to its own private copy of count.',
        why: 'The pointer count is private to each thread, like i. The int it points to is one location in global memory, shared by every thread.',
      },
    ],
    explain: (
      <>
        <p>
          The chapter's test for one thread per output fails here: there is one output, and 300 threads compute it
          from what the others wrote. No order between the threads is promised, inside a block or between blocks, so
          updates can be lost. The cures are atomic operations (chapter 9) and a reduction (chapter 10), which is also
          how the sum behind a softmax is computed.
        </p>
        <p>
          The course's emulator shows why you cannot test your way out of this. In one-off runs of this file made
          when this page was written, the <code>emu</code> flavour, which runs one thread at a time and never
          interrupts an update, gave 300 every time. The <code>race</code> flavour, whose threads run at once, gave
          188, 191 and 202 in three runs, and ThreadSanitizer named the line of the update. Neither is a GPU.
        </p>
      </>
    ),
    source: 'our kernel; criterion of ch. 2 §2.1, pp. 21–23; no order between blocks, §2.6, p. 39, nor between the threads of a block, ch. 4 §4.4, p. 74; private variables, §2.5, p. 37',
  },
  'ch02/wrap/q-bytes-int': {
    kind: 'choice',
    id: 'ch02/wrap/q-bytes-int',
    prompt: (
      <>
        You copy the stub of Fig. 2.8, with its <code>int size = n * sizeof(float);</code>, to put an embedding table
        on the device: {n(TABLE.rows)} tokens by {n(TABLE.cols)} dimensions of <code>float</code>, so{' '}
        <code>int n = {n(TABLE_N)}</code>. What goes wrong?
      </>
    ),
    options: [
      {
        label: 'Nothing: sizeof makes the product a size_t, which is large enough.',
        why: 'The product is computed as a size_t and is right. It is then stored in an int, and that is where it stops being right.',
      },
      {
        label: `size cannot hold ${n(TABLE_BYTES)}, so it does not hold the byte count, and the allocation and both copies ask for the wrong size.`,
        correct: true,
      },
      {
        label: 'The kernel\'s int i overflows for the last elements.',
        why: `The largest index is ${n(TABLE_N - 1)}, well below ${n(INT_MAX)}. An index into a table of this size still fits in an int; its size in bytes does not.`,
      },
      {
        label: 'The line no longer compiles.',
        why: 'Converting a size_t to an int is a narrowing conversion that C++ allows without a word. Some compilers warn about it when asked to.',
      },
    ],
    explain: (
      <p>
        {n(TABLE_N)} floats are {n(TABLE_BYTES)} bytes. The largest value of a 32-bit <code>int</code> is{' '}
        {n(INT_MAX)}. Since C++20 the conversion is defined to wrap modulo 2³², so <code>size</code> holds{' '}
        {n(WRAPPED)}, and when that is passed where <code>cudaMalloc</code> expects a <code>size_t</code> it becomes an
        enormous unsigned number. The request is wrong before the runtime ever sees it. Fig. 2.8 is fine for the
        book's vectors; the labs write <code>size_t size = (size_t)n * sizeof(float);</code>, and lab 2.1's solution
        notes that an <code>int</code> byte count overflows once n reaches 2²⁹ floats.
      </p>
    ),
    source: 'ch. 2 Fig. 2.8, p. 31; int versus unsigned, footnote 4, p. 37; the conversion rule is C++\'s, ours',
  },
  'ch02/wrap/q-overlap': {
    kind: 'choice',
    id: 'ch02/wrap/q-overlap',
    prompt: (
      <>
        A host function launches a kernel that runs for 3 ms, then spends 2 ms of CPU time tokenising the next
        request, then calls <code>cudaMemcpy</code> to fetch the kernel's result. Leaving out the launch overhead and
        the copy itself, about how long after the launch does <code>cudaMemcpy</code> return? (The times are invented
        for this question.)
      </>
    ),
    options: [
      {
        label: 'About 5 ms: the CPU work starts once the kernel has finished.',
        why: 'The launch returns before the kernel has run, so the CPU work starts at once.',
      },
      { label: 'About 3 ms: the CPU work runs while the kernel runs, and the copy waits for the last 1 ms.', correct: true },
      {
        label: 'About 2 ms: the copy starts right after the CPU work and fetches whatever is there.',
        why: 'The copy goes into the same in-order default stream as the kernel and does not return until it has completed, so it comes after the kernel.',
      },
      {
        label: 'It cannot be said: without cudaDeviceSynchronize nothing orders the copy after the kernel.',
        why: 'The default stream does: work in it runs in the order it was issued.',
      },
    ],
    explain: (
      <p>
        This is the overlap the book's Fig. 2.3 hints at and never states. NVIDIA documents three things: a launch
        returns before the kernel completes, work issued without a stream runs in order in the default stream, and{' '}
        <code>cudaMemcpy</code> does not return until the copy is done. That the copy therefore waits for the kernel is
        our reading of those three statements; no single sentence says it. On Metal, <code>commit</code> returns at
        once in the same way, and <code>waitUntilCompleted</code> is the wait. An inference server uses exactly this
        to prepare the next request while the GPU works.
      </p>
    ),
    source: 'ch. 2 §2.2, p. 24, Fig. 2.3; notes/today/ch02-facts.md, A5 (the inference is ours) and C4',
  },
};

const M0 = 'ch02/lab3_elementwise';
// Backends that are not a GPU: the emulator and its race-detection flavour.
const EMULATED = new Set(['emu', 'race']);

export function Wrap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const devices = Object.keys(reports).filter((b) => !EMULATED.has(b));
  const m0 = devices.filter((b) => labStatus(reports[b], M0, 'work') === 'pass');
  const m0emu = labStatus(reports.emu, M0, 'work') === 'pass';
  const ids = Object.keys(progress.answers).filter((k) => k.startsWith('ch02/'));
  const right = ids.filter((k) => progress.answers[k].correct).length;
  const asked = ids.filter((k) => progress.answers[k].correct !== undefined).length;
  return (
    <>
      <h2 style={{ marginTop: 0 }}>What to carry forward</h2>
      <ul>
        <li>
          An operation is data parallel when no output is computed from another output. Then it can have one thread
          per output, and no order of the threads changes the result.
        </li>
        <li>
          A program is one host thread that launches grids of device threads, organised in blocks. A launch returns
          before its grid has run. Waiting is something you ask for: <code>cudaDeviceSynchronize()</code>, a blocking
          call in the same stream, or <code>waitUntilCompleted</code> on Metal.
        </li>
        <li>
          A host stub has three parts: allocate and copy in, launch, copy back and free. Every crossing of the
          boundary costs, which is why real code keeps its data on the device. Lab 2.2 measures by how much.
        </li>
        <li>
          Sizes are in bytes, the destination comes first, and on CUDA a device pointer goes only to API calls and
          kernels. CUDA does not clear the memory it allocates.
        </li>
        <li>
          Each thread finds its element with <code>i = blockIdx.x * blockDim.x + threadIdx.x</code> and tests{' '}
          <code>i &lt; n</code>. Its automatic variables are its own. The grid takes the place of the loop.
        </li>
        <li>
          The block count is <code>(n + B - 1) / B</code> in integers. A launch statement returns no error code: read
          the error state right after it, then check the wait. Blocks run in any order.
        </li>
        <li>
          Qualifiers decide who compiles a function, who may call it and who runs it. One{' '}
          <code>__host__ __device__</code> definition is built twice. Metal has no such function: its kernels are a
          separate language, finished for your GPU when the pipeline is created.
        </li>
      </ul>
      <Source section="2.8" pages="40" />
      <p>
        The book ends the chapter by sorting what it introduced into four groups of extensions. Here they are with
        their counterparts on the other two backends.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, B1, B4, C1 to C5, C7)">
        <p>
          HIP keeps CUDA's spelling once <code>&lt;hip/hip_runtime.h&gt;</code> is included. A shared-storage Metal
          buffer is read and written through <code>contents</code> and is released with its last reference.
        </p>
      </Today>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-table="four-groups">
          <thead>
            <tr><th>The book's group</th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>Function declarations</td>
              <td><code>__global__</code>, <code>__device__</code>, <code>__host__</code></td>
              <td>the same three</td>
              <td><code>kernel</code>; any other function is GPU code; nothing compiles one function for both sides</td>
            </tr>
            <tr>
              <td>Kernel call and grid launch</td>
              <td><code>kernel&lt;&lt;&lt;blocks, threads&gt;&gt;&gt;(args)</code></td>
              <td>the same syntax</td>
              <td>an encoder: pipeline, buffers, <code>dispatchThreads</code> or <code>dispatchThreadgroups</code>, then <code>commit</code></td>
            </tr>
            <tr>
              <td>Built-in variables</td>
              <td><code>threadIdx</code>, <code>blockIdx</code>, <code>blockDim</code></td>
              <td>the same names</td>
              <td>kernel arguments with attributes, such as <code>[[thread_position_in_grid]]</code></td>
            </tr>
            <tr>
              <td>Runtime API</td>
              <td><code>cudaMalloc</code>, <code>cudaMemcpy</code>, <code>cudaFree</code></td>
              <td><code>hipMalloc</code>, <code>hipMemcpy</code>, <code>hipFree</code></td>
              <td><code>newBufferWithLength</code>, <code>newBufferWithBytes</code>, <code>contents</code>; no free call</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>A mixed check</h2>
      <p>
        These draw on the whole chapter and on situations the sections did not show.
        {asked ? ` So far in this chapter you have answered ${right} of ${asked} questions correctly, counting your latest answer to each.` : ''}
      </p>
      <Code
        tabs={[
          {
            label: 'CUDA (not a lab file: for this question only)',
            code: SAXPY,
            tiers: { x: 'global', y: 'global', x_d: 'global', y_d: 'global', i: 'reg' },
            status: 'front-end-checked; has a defect on purpose',
          },
        ]}
        caption={
          <>
            Ours, not the book's. The checked copy is <code>labs/ch02/snippets/saxpy_stub_defect.cu</code>.
          </>
        }
      />
      <Ask q={questionsWrap['ch02/wrap/q-inout']} label="Find the defect" />
      <Ask q={questionsWrap['ch02/wrap/q-logits']} />
      <Code
        tabs={[
          {
            label: 'CUDA (not a lab file: for this question only)',
            code: COUNT,
            tiers: { p: 'global', count: 'global', i: 'reg' },
            status: 'front-end-checked; has a defect on purpose',
          },
        ]}
        caption={
          <>
            Ours, not the book's. The checked copy is <code>labs/ch02/snippets/count_above_race.cu</code>.
          </>
        }
      />
      <Ask q={questionsWrap['ch02/wrap/q-count']} />
      <Ask q={questionsWrap['ch02/wrap/q-bytes-int']} />
      <Ask q={questionsWrap['ch02/wrap/q-overlap']} />

      <h2>The book's exercises</h2>
      <p>
        The chapter ends with ten exercises (pp. 41–42). Each is worked somewhere in this course; the book prints no
        answers, so the answers here are ours.
      </p>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-table="exercises">
        <thead>
          <tr><th>Exercise</th><th>About</th><th>Where it is worked here</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>1</td>
            <td>The index expression for one element per thread</td>
            <td>a question in <a href={href('ch02', '2.5')}>2.5</a></td>
          </tr>
          <tr>
            <td>2, 3</td>
            <td>Two elements per thread, adjacent or in two sections</td>
            <td>two questions in <a href={href('ch02', '2.5')}>2.5</a>, which unlock those mappings in its index explorer</td>
          </tr>
          <tr>
            <td>4</td>
            <td>Threads in the grid for 8000 elements in blocks of 1024</td>
            <td>a question in <a href={href('ch02', '2.6')}>2.6</a>, and its block-count calculator</td>
          </tr>
          <tr>
            <td>5, 6</td>
            <td>The two arguments of <code>cudaMalloc</code></td>
            <td>two questions in <a href={href('ch02', '2.4')}>2.4</a></td>
          </tr>
          <tr>
            <td>7</td>
            <td>The <code>cudaMemcpy</code> call for a copy to the device</td>
            <td>a question in <a href={href('ch02', '2.4')}>2.4</a>, with four calls of our own</td>
          </tr>
          <tr>
            <td>8</td>
            <td>The type that receives an API call's result</td>
            <td>
              the question on <code>cudaMalloc</code>'s out-parameter in <a href={href('ch02', '2.4')}>2.4</a>, and the{' '}
              <code>CHECK</code> macro you write in lab 2.1 (<a href={href('ch02', 'labs')}>labs</a>)
            </td>
          </tr>
          <tr>
            <td>9</td>
            <td>Blocks, threads and working threads of Fig. 2.15</td>
            <td>"Count the threads" in <a href={href('ch02', '2.6')}>2.6</a></td>
          </tr>
          <tr>
            <td>10</td>
            <td>A programmer who writes every function twice</td>
            <td>a question in <a href={href('ch02', '2.7')}>2.7</a>; lab 2.3's <code>scaleShift</code> does it right</td>
          </tr>
        </tbody>
      </table>

      <h2>Where the PDF looks wrong</h2>
      <p>
        This edition is new and the copy this course was built from has apparent slips in chapter 2. The first three
        are in code: as printed, two of them do not compile and the third invites the integer-division bug. That is
        why the lessons show only tested files and never retype a listing. The rest are listed so that you are not
        thrown by them when you read the book alongside.
      </p>
      <ul data-list="errata">
        <li>
          Fig. 2.5, line 03 (p. 27): <code>float *d_A *d_B, *d_C;</code> lacks a comma after <code>*d_A</code>, and uses
          a <code>d_</code> prefix where everything else uses the <code>_d</code> suffix.{' '}
          <a href={href('ch02', '2.4')}>Section 2.4</a> has you find it.
        </li>
        <li>
          The error check on p. 32 declares <code>cudaError_t err</code> and then tests <code>error</code>. Section
          2.4 has you find it too, and lab 2.1's <code>CHECK</code> macro is the corrected version.
        </li>
        <li>
          Fig. 2.12 (p. 38) names the function <code>vectAdd</code>, gives it return type <code>int</code> with no{' '}
          <code>return</code>, and its comment says <code>ceil(n/256)</code>, which read as C++ with an integer{' '}
          <code>n</code> is the bug <a href={href('ch02', '2.6')}>section 2.6</a> takes apart. Everywhere else the
          function is <code>void vecAdd</code>.
        </li>
        <li>
          The captions of Figs. 2.12 and 2.13 (p. 38) read as if swapped: the one that shows only the launch is called
          a host code example, and the complete host function is called a kernel call statement.
        </li>
        <li>
          The <code>_h</code> and <code>_d</code> naming of p. 26 is not kept by the listings. Figs. 2.4 and 2.8 use{' '}
          <code>A_h</code>, <code>B_h</code>, <code>C_h</code>; Figs. 2.5, 2.12 and 2.13 use plain <code>A</code>,{' '}
          <code>B</code>, <code>C</code> for the host arrays. Inside the kernel the text drops the suffixes (p. 35),
          as Fig. 2.10 does, but the code box of Fig. 2.9 writes <code>A_d</code>, <code>B_d</code>, <code>C_d</code>.
        </li>
        <li>The code box of Fig. 2.9 (p. 33) has no <code>if (i &lt; n)</code>; Fig. 2.10 has it.</li>
        <li>
          p. 37 says the kernel of "Fig. 2.10" has no loop corresponding to the one in Fig. 2.10; the loop is in Fig.
          2.4. The same page calls the kernel <code>vecAddVecKernel</code>; it is <code>vecAddKernel</code>.
        </li>
        <li>The snippet on p. 30 prints <code>float *A_d</code> without its semicolon.</li>
        <li>
          Exercise 3 (p. 41) reads "will first process a section first", and prints the same four options as exercise
          2. The intended answers differ: (c) for exercise 2, (d) for exercise 3, by our derivation.
        </li>
        <li>
          Smaller things: "discussed in 5" on p. 28 is missing the word Chapter; a sentence on p. 36 is missing the
          word "in"; footnote 2 (p. 29) says <code>cudaMalloc</code> "returns a generic object", where the text explains
          that it writes the pointer through its first parameter; reference [1] (p. 43) gives three years for one
          book.
        </li>
      </ul>
      <Aside>
        <p>These are our readings of one PDF. If your copy differs, trust your copy.</p>
      </Aside>
      <p>
        Two things look like slips and are not. The text and Fig. 2.9 write the index as{' '}
        <code>blockIdx.x * blockDim.x + threadIdx.x</code> and Fig. 2.10 as{' '}
        <code>threadIdx.x + blockDim.x * blockIdx.x</code>: the same value. And two options of exercise 5 use an{' '}
        <code>n</code> the question never defines; they are distractors.
      </p>

      <h2>Where today's documentation differs from the page</h2>
      <p>
        The book's reference for CUDA is NVIDIA's programming guide of release 13.0 (its reference [3]). The vendors'
        current pages add to the chapter or say things differently in these places, each taken up in the section
        named.
      </p>
      <ul data-list="today">
        <li>
          A launch is asynchronous: it returns before the kernel has run. The book never says so in this chapter; its
          listings work because the blocking copy back runs after the kernel in the default stream (
          <a href={href('ch02', '2.2')}>2.2</a>, <a href={href('ch02', '2.6')}>2.6</a>, lab 2.2).
        </li>
        <li>
          A launch statement returns no error code, and the book's check does not cover it. NVIDIA's pattern is{' '}
          <code>cudaGetLastError()</code> right after the launch and a checked <code>cudaDeviceSynchronize()</code>{' '}
          after that (<a href={href('ch02', '2.6')}>2.6</a>, lab 2.1).
        </li>
        <li>
          NVIDIA teaches the block count as the integer <code>(n + threads - 1) / threads</code>, or{' '}
          <code>cuda::ceil_div</code>, not <code>ceil(n/256.0)</code>. What CUDA does with a grid of zero blocks is
          not documented; HIP returns an error (<a href={href('ch02', '2.6')}>2.6</a>).
        </li>
        <li>
          The <code>(void**)</code> cast on <code>cudaMalloc</code> is no longer required: C++ code can use a typed
          overload. The cast still compiles everywhere, and the labs keep it (<a href={href('ch02', '2.4')}>2.4</a>).
        </li>
        <li>
          There is a fifth copy kind, <code>cudaMemcpyDefault</code>, which infers the direction from the pointers and
          which NVIDIA now recommends (<a href={href('ch02', '2.4')}>2.4</a>).
        </li>
        <li>
          The qualifier table has grown two rows, <code>__tile__</code> and <code>__tile_global__</code>, for CUDA
          Tile kernels. The book's three rows are unchanged (<a href={href('ch02', '2.7')}>2.7</a>).
        </li>
        <li>
          By default <code>nvcc</code> compiles device code to machine code at build time and packs it with PTX. The
          driver, not a part of NVCC, compiles the PTX at run time, and only when no machine code fits the GPU (
          <a href={href('ch02', '2.7')}>2.7</a>).
        </li>
        <li>
          Unified Virtual Addressing, in NVIDIA's current wording, is about addresses. What lets a kernel read host
          memory is mapped or unified memory (<a href={href('ch02', '2.5')}>2.5</a>).
        </li>
        <li>
          The guide of release 13.0 is now an archive; the current guide opens with this chapter's vector addition.
          The book's device figures are its own illustrations: "one or two" and "128 or 256" blocks at once have no
          source, and the H100's 80 GB and 94 GB are two products.
        </li>
      </ul>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md, A1, A2, A4 to A8, A12, A14, A15, A18)">
        <p>
          Checked against NVIDIA's programming guide v13.5 and runtime API reference, AMD's HIP documentation and
          Apple's Metal documentation. The book's claims on thread-creation cycles and on CUDA being the most mature
          framework could not be checked either way.
        </p>
      </Today>

      <h2>What this chapter left out</h2>
      <ul>
        <li>
          From section 2.1: the list of data-heavy applications (p. 21), the photograph of Fig. 2.1, most of the RGB
          sidebar (p. 22) and the molecular-dynamics task list of the data-versus-task sidebar (p. 23). The grayscale
          kernel itself is chapter 3.
        </li>
        <li>The sidebar on pointers in C++ (pp. 26–27).</li>
        <li>
          Multidimensional arrays with <code>cudaMalloc</code> (footnote 2, p. 29): chapter 3. The faster or copy-free
          allocation functions the book defers (p. 29) are named in section 2.4 from today's documentation; the rest is
          chapter 23 and appendix C.
        </li>
        <li>
          The telephone-number analogy of the hierarchy sidebar (pp. 34–35) and the <code>int</code> versus{' '}
          <code>unsigned</code> footnote (p. 37) are one sentence each; the UVA footnote (p. 35) is a margin note, with
          the detail in appendix C. Dynamic parallelism (p. 35) gets one paragraph in section 2.7: this edition
          removed its chapter.
        </li>
        <li>
          Fig. 2.14 is replaced by a pipeline for each of the three backends that follows today's documentation.
        </li>
        <li>
          The coarsened vector addition of exercises 2 and 3 is worked as index arithmetic, not as a lab. Coarsening is
          measured properly in chapters 6 and 15.
        </li>
        <li>
          Named and not taught here: <code>cudaMemcpyDefault</code>, streams and events,{' '}
          <code>cudaLaunchKernelEx</code>, CUDA Tile kernels, Metal 4's command types, Metal private storage and blit
          copies, and the run-time compilers hiprtc and nvrtc.
        </li>
      </ul>

      <h2>The project so far</h2>
      <p>
        Lab 2.3's kernels, bias add, scale and shift, and residual add, with a launch function beside each, are
        milestone M0 of the through-line project: the elementwise operations every block of the small transformer you
        will end up running needs.{' '}
        <span data-m0>
          {devices.length === 0 && !reports.emu
            ? 'No lab results are recorded on this machine yet.'
            : m0.length
              ? `Your own version passes on ${m0.join(' and ')} on this machine.`
              : m0emu
                ? 'Your own version passes on the emulator; it has not passed on a device on this machine yet.'
                : 'Your own version has not passed on this machine yet.'}
        </span>{' '}
        See the <a href={href('ch02', 'labs')}>labs</a>.
      </p>
      <Beyond>
        <p>The project, <code>nanoinfer</code>, and its milestones are ours, not the book's.</p>
      </Beyond>
      <p>
        <a href={href('ch03', 'start')}>Chapter 3</a> takes the grid from one dimension to two and three. It writes
        the grayscale kernel that section 2.1 used as its example, then a blur, then the matrix multiplication behind
        a linear layer, which is the next milestone, M1. Everything in this chapter's host stub carries over
        unchanged; what changes is how a thread finds its element.
      </p>
    </>
  );
}
