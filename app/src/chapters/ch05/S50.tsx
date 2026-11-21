import { Ask, type Question } from '../../components/Ask';
import { Aside, Source } from '../../components/notes';
import { Code, MATMUL_TIERS } from '../../components/Code';
import { href } from '../../lib/router';

// The kernel of the book's Fig. 3.11 as the labs use it: labs/ch05/shared/naive_matmul.cu
// (snippets.test.ts keeps the two identical). It differs from the page only by `const`
// on the inputs and the kernel's name.
const NAIVE = `__global__ void naiveMatmulKernel(const float* M, const float* N, float* P, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < Width && col < Width) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }
        P[row * Width + col] = Pvalue;
    }
}`;

// Prerequisite questions drawn from chapters 3 and 4 (from the notes for those chapters).
export const questions50: Record<string, Question> = {
  'ch05/start/q-rowmajor': {
    kind: 'numeric',
    id: 'ch05/start/q-rowmajor',
    prompt: (
      <>
        A matrix with <code>Width = 6</code> is stored row-major in a flat array <code>N</code>. The kernel above walks
        down <b>column 2</b> of <code>N</code>. By how many array elements does the address move between consecutive
        iterations of <code>k</code>?
      </>
    ),
    answer: 6,
    unit: 'elements',
    explain: (
      <p>
        <code>N[k*Width + col]</code>: each step of <code>k</code> adds one whole row, so the stride is{' '}
        <code>Width</code>. The read of <code>M[row*Width + k]</code> moves by one. Keep this asymmetry in mind: M is
        read along memory, N across it. Chapter 6 turns it into a performance fact.
      </p>
    ),
    source: 'ch. 3 §3.4, p. 62',
  },
  'ch05/start/q-mapping': {
    kind: 'choice',
    id: 'ch05/start/q-mapping',
    prompt: (
      <>
        The naive kernel is launched with 16×16 blocks on a 40×40 matrix. Which threads of the block at{' '}
        <code>blockIdx = (x 2, y 0)</code> do real work?
      </>
    ),
    options: [
      {
        label: 'All 256: every thread of a launched block runs the kernel body.',
        why: 'Every thread runs the body, but only those whose column exists pass the bounds test. This block covers columns 32 to 47, and the matrix stops at column 39.',
      },
      {
        label: 'The 128 with threadIdx.x from 0 to 7.',
        correct: true,
        why: 'Columns 32 to 39 exist; rows 0 to 15 all exist.',
      },
      {
        label: 'The 128 with threadIdx.y from 0 to 7: the matrix ends 8 rows into this block.',
        why: 'x maps to columns here, y to rows. This block has blockIdx.y = 0, so its rows 0 to 15 all exist.',
      },
      {
        label: 'The 64 with threadIdx.x and threadIdx.y both from 0 to 7.',
        why: 'That is the corner block, x 2 and y 2. This block sits in the top row of the grid, where every row exists.',
      },
    ],
    explain: (
      <p>
        <code>col = blockIdx.x*blockDim.x + threadIdx.x = 32 + threadIdx.x</code> must be below 40, so{' '}
        <code>threadIdx.x ≤ 7</code>. <code>row = threadIdx.y</code> is always below 40. The other 128 threads fail the{' '}
        <code>if</code> and exit. Later in this chapter you will meet threads that own no output element and still have
        work to do.
      </p>
    ),
    source: 'ch. 3 §3.2 and §3.4, pp. 48–54, 61',
  },
  'ch05/start/q-barrier': {
    kind: 'choice',
    id: 'ch05/start/q-barrier',
    prompt: (
      <>
        What does <code>__syncthreads()</code> guarantee?
      </>
    ),
    options: [
      {
        label: 'No thread of its block passes it until all threads of that block reach it.',
        correct: true,
      },
      {
        label: 'No thread of the grid proceeds until every thread of the grid has reached it.',
        why: 'The scope is one block. Blocks cannot wait for each other this way, which is what lets them run in any order.',
      },
      {
        label: 'The threads of the block then execute the statements after it one at a time, like a lock.',
        why: 'It is a barrier, not a lock: after it, threads run concurrently again.',
      },
      {
        label: 'Every global-memory write made before it becomes visible to every thread of the grid.',
        why: 'Its scope is one block. Threads of other blocks are not ordered by it, so it promises them nothing.',
      },
    ],
    explain: (
      <p>
        It is a barrier scoped to one block. Two consequences you will use today: every thread of the block must execute
        the same barrier, and a barrier says nothing about other blocks.
      </p>
    ),
    source: 'ch. 4 §4.3, pp. 70–72',
  },
  'ch05/start/q-occupancy': {
    kind: 'numeric',
    id: 'ch05/start/q-occupancy',
    prompt: (
      <>
        In the book's example device, a streaming multiprocessor can hold 2048 threads and has 65,536 registers. A
        kernel compiles to 40 registers per thread. What is the best occupancy it can reach, in percent? Ignore the
        block-size granularity.
      </>
    ),
    answer: 80,
    tol: 0.01,
    unit: '%',
    explain: (
      <p>
        65,536 / 40 = 1638 threads fit, and 1638 / 2048 = 80%. Registers are one of three budgets that cap how many
        threads an SM can hold; this chapter adds a second, shared memory. (With real block sizes the figure is lower
        still: only whole blocks fit.)
      </p>
    ),
    source: 'ch. 4 §4.7, pp. 85–87 (device figures are the book\'s Hopper H100 example)',
  },
};

export function S50() {
  return (
    <>
      <p>
        When you call <code>torch.matmul</code> on a GPU tensor, the framework hands the work to a matrix
        multiplication routine tuned for that device. This chapter is where you start to see why such routines look
        nothing like the three nested loops you would write first, and you write the first real improvement yourself.
      </p>
      <Source pages="93" />
      <p>
        The kernel from chapter 3 is correct and uses thousands of threads, yet on the book's reference GPU it reaches
        about one percent of the arithmetic the hardware can do. The threads are not short of work. They are waiting
        for memory. Everything in this chapter follows from that one observation:
      </p>
      <ul>
        <li>
          a way to <b>measure</b> it: operations done per byte fetched, and a picture (the roofline) that tells you
          which limit a kernel is up against;
        </li>
        <li>
          the <b>places data can live</b> on a GPU, and what each costs;
        </li>
        <li>
          a technique, <b>tiling</b>, that moves the data a block of threads keeps reusing into memory that is on the
          chip;
        </li>
        <li>
          and the <b>bill</b> for that technique: barriers, bounds checks and a memory budget.
        </li>
      </ul>
      <Aside>
        <p>
          <b>Colour in this course</b> means where data lives:{' '}
          <span className="chip reg">register</span> <span className="chip shared">shared memory</span>{' '}
          <span className="chip global">global memory</span> <span className="chip const">constant memory</span>. The
          same colours mark variables in code. Red always means something failed.
        </p>
      </Aside>

      <h2>The kernel you are about to take apart</h2>
      <p>
        One thread computes one element of <code>P = M × N</code>. Read it with the colours: everything inside the loop
        that is not a register is a trip to global memory.
      </p>
      <Source section="3.4" pages="61" figs="Fig. 3.11" />
      <Code
        tabs={[{ label: 'CUDA', code: NAIVE, tiers: MATMUL_TIERS, status: 'front-end-checked (labs/ch05/shared/naive_matmul.cu)' }]}
        highlight={[7]}
        caption="The naive kernel of the book's Fig. 3.11, in the form the labs use as their baseline (the book names it MatrixMulKernel and leaves out const). Line 7 is the whole story of this chapter: two loads from global memory for one multiply and one add."
      />

      <h2>Before you start</h2>
      <p>
        Four questions on what this chapter assumes from chapters 3 and 4. They are worth getting wrong now rather than
        later; each one comes back in <a href={href('review')}>review</a>.
      </p>
      <Ask q={questions50['ch05/start/q-rowmajor']} label="From chapter 3" />
      <Ask q={questions50['ch05/start/q-mapping']} label="From chapter 3" />
      <Ask q={questions50['ch05/start/q-barrier']} label="From chapter 4" />
      <Ask q={questions50['ch05/start/q-occupancy']} label="From chapter 4" />
    </>
  );
}
