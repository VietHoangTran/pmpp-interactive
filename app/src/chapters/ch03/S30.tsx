import { Ask, type Question } from '../../components/Ask';
import { Aside, Source, type Tier } from '../../components/notes';
import { Code, useLabFile } from '../../components/Code';
import { href } from '../../lib/router';
import { coverage, dim3, globalCoord, sizeGrid } from '../../sim/grid';

const VEC_TIERS: Record<string, Tier> = {
  A: 'global', B: 'global', C: 'global', A_d: 'global', B_d: 'global', C_d: 'global',
  i: 'reg', n: 'reg', threads: 'reg', blocks: 'reg',
};

// The numbers the prerequisite questions ask for, from the same model section 3.1 runs on.
const INDEX = globalCoord(dim3(3, 0, 0), dim3(256), dim3(10, 0, 0)).x;
const EX4 = coverage(8000, 1024, sizeGrid(8000, 1024, 'round-up'));

// Prerequisite questions drawn from chapter 2 (from the notes for that chapter).
export const questionsS30: Record<string, Question> = {
  'ch03/start/q-globalindex': {
    kind: 'numeric',
    id: 'ch03/start/q-globalindex',
    prompt: (
      <>
        The kernel above is launched with 256 threads in each block. Which element does the thread with{' '}
        <code>blockIdx.x = 3</code> and <code>threadIdx.x = 10</code> handle? Give the value of <code>i</code>.
      </>
    ),
    answer: INDEX,
    explain: (
      <p>
        <code>i = blockIdx.x * blockDim.x + threadIdx.x</code> = 3 × 256 + 10 = {INDEX}. Blocks 0, 1 and 2 cover
        elements 0 to 767, so block 3 starts at 768. This chapter uses the same formula once for each dimension: one
        copy for the column, one for the row.
      </p>
    ),
    source: 'ch. 2 §2.5, p. 34',
  },
  'ch03/start/q-surplus': {
    kind: 'numeric',
    id: 'ch03/start/q-surplus',
    prompt: (
      <>
        A vector of 8000 elements, one element per thread, blocks of 1024 threads, and the fewest blocks that cover
        it. How many threads fail the <code>i &lt; n</code> test?
      </>
    ),
    answer: EX4.surplus,
    unit: 'threads',
    explain: (
      <p>
        8000 / 1024 rounds up to {EX4.blocks} blocks, which is {EX4.threads.toLocaleString('en-US')} threads.{' '}
        {EX4.working.toLocaleString('en-US')} of them have an element; the other {EX4.surplus} compute an index past
        the end and must do nothing. A grid is always a whole number of blocks, so some surplus is normal. In two
        dimensions it becomes whole rows and columns of threads, and the test gets a second half.
      </p>
    ),
    source: 'ch. 2 §2.5 and §2.6, pp. 37–39; exercise 4, p. 41',
  },
  'ch03/start/q-bytes': {
    kind: 'choice',
    id: 'ch03/start/q-bytes',
    prompt: (
      <>
        You allocate device memory for a colour image, <code>w</code> pixels wide and <code>h</code> high, stored as
        three <code>unsigned char</code> per pixel. What is the size argument of <code>cudaMalloc</code>?
      </>
    ),
    options: [
      {
        label: <code>w * h * 3</code>,
        correct: true,
        why: 'The size is in bytes, and an unsigned char is one byte.',
      },
      { label: <code>w * h</code>, why: 'That is the number of pixels. Each pixel takes three bytes.' },
      {
        label: <code>w * h * 3 * sizeof(float)</code>,
        why: 'Four times too much on a machine with 4-byte floats: the elements are unsigned char, not float.',
      },
      {
        label: <code>(void**)&amp;Pin_d</code>,
        why: 'That is the first argument: the address of the pointer that will receive the allocation.',
      },
    ],
    explain: (
      <p>
        <code>cudaMalloc</code> takes the address of a pointer, cast to <code>(void**)</code>, and a size in{' '}
        <b>bytes</b>. It writes the pointer through its first argument so that its return value is free to report an
        error. The first kernel of this chapter reads exactly such a buffer, and the factor of 3 turns up again inside
        it as an index.
      </p>
    ),
    source: 'ch. 2 §2.4, pp. 28–30',
  },
  'ch03/start/q-order': {
    kind: 'choice',
    id: 'ch03/start/q-order',
    prompt: <>A kernel is launched over many blocks. May its code rely on block 0 finishing before block 1 starts?</>,
    options: [
      {
        label: 'No. Blocks may execute in any order.',
        correct: true,
      },
      {
        label: 'Yes. Blocks run in increasing order of blockIdx.x.',
        why: 'The book promises no order at all between blocks.',
      },
      {
        label: 'Only on a small GPU that runs one block at a time.',
        why: 'A small GPU may run one or two blocks at a time, but which one goes first is still not promised.',
      },
      {
        label: 'Yes, as long as the grid is 1D.',
        why: 'The shape of the grid has no bearing on the order in which its blocks run.',
      },
    ],
    explain: (
      <p>
        Blocks work on different parts of the data and may run in any order, some of them at the same time. That is
        why every kernel in this chapter has the same discipline: a thread writes only its own output element and
        reads only inputs that no thread writes. With no shared writes, no order can change the result.
      </p>
    ),
    source: 'ch. 2 §2.6, p. 39',
  },
};

export function S30() {
  const vecAdd = useLabFile('ch03/snippets/vec_add_1d.cu');
  return (
    <>
      <p>
        In PyTorch a tensor is a shape and a flat buffer. When you write <code>x[b, t, h]</code> or call{' '}
        <code>torch.matmul</code>, two pieces of arithmetic happen out of sight. Something decides which unit of work
        handles which element, and something turns the indices <code>(b, t, h)</code> into one offset into the buffer.
        A kernel author writes both by hand. This chapter is those two pieces of arithmetic.
      </p>
      <Source pages="45" />

      <h2>What chapter 2 left you with</h2>
      <p>
        One kernel call that launches a one-dimensional grid of threads over one-dimensional arrays. Each thread
        builds a single global index from three built-in variables, tests it against the length, and handles one
        element.
      </p>
      <Code
        tabs={[{ label: 'CUDA', code: vecAdd ?? '// loading…', status: 'front-end-checked', tiers: VEC_TIERS }]}
        highlight={[4, 5]}
        caption="Chapter 2 in two functions. Line 4 is the mapping from a thread to its element and line 5 the test that stops surplus threads. Everything in this chapter is these two lines in more dimensions."
      />

      <h2>What this chapter adds</h2>
      <ul>
        <li>
          <b>Grids and blocks of up to three dimensions</b>, set by two <code>dim3</code> values at the launch, and a
          fourth built-in variable, <code>gridDim</code>, beside the three you know (
          <a href={href('ch03', '3.1')}>section 3.1</a>).
        </li>
        <li>
          <b>Row-major flattening</b>: where element <code>[row][col]</code> sits in a flat array, which is how every
          array in this chapter reaches its kernel (<a href={href('ch03', '3.2')}>section 3.2</a>).
        </li>
        <li>
          <b>Three kernels of rising complexity.</b> Colour to grayscale is pure mapping: one thread, one pixel, no
          loop. Blur gives each thread a loop over a neighbourhood and a second boundary test inside it (
          <a href={href('ch03', '3.3')}>section 3.3</a>). Matrix multiplication gives each thread a whole dot product
          (<a href={href('ch03', '3.4')}>section 3.4</a>).
        </li>
      </ul>
      <p>
        Be clear about what the matrix multiplication here is for. Chapter 5 takes that kernel apart for performance.
        This chapter only gets it correct: every thread writes the right element from the right inputs, for any size.
        Do not judge its speed yet.
      </p>
      <Aside>
        <p>
          <b>Colour in this course</b> means where data lives. This chapter uses two of the four:{' '}
          <span className="chip reg">register</span> for a thread's private variables and{' '}
          <span className="chip global">global memory</span> for the arrays. The same colours mark variables in code.
          Red always means something failed.
        </p>
      </Aside>

      <h2>What you will do</h2>
      <ul>
        <li>Lay out a launch in one, two or three dimensions and say which shapes a device rejects.</li>
        <li>Step one thread from its four indices to the byte it reads.</li>
        <li>Find three errors in a kernel listing as the book prints it.</li>
        <li>
          Work through <a href={href('ch03', 'labs')}>five labs</a> on your own GPU: grayscale, blur, a naive GEMM for
          a linear layer, one thread per element against one per row or column, and a hunt for four index defects.
          The third is the first half of milestone M1 of the course project.
        </li>
      </ul>

      <h2>A check on chapter 2</h2>
      <p>
        Four questions on what this chapter assumes from chapter 2. They are worth getting wrong now rather than
        later; each one comes back in <a href={href('review')}>review</a>.
      </p>
      <Ask q={questionsS30['ch03/start/q-globalindex']} label="From chapter 2" />
      <Ask q={questionsS30['ch03/start/q-surplus']} label="From chapter 2" />
      <Ask q={questionsS30['ch03/start/q-bytes']} label="From chapter 2" />
      <Ask q={questionsS30['ch03/start/q-order']} label="From chapter 2" />
    </>
  );
}
