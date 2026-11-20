import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source, type Tier } from '../../components/notes';
import { Code, useLabFile } from '../../components/Code';
import { href } from '../../lib/router';
import { ceiling, speedup } from '../../sim/amdahl';

const LAB1 = 'ch02/lab1_vecadd/solution/kernel.cu';

const VEC_TIERS: Record<string, Tier> = { A: 'global', B: 'global', C: 'global', i: 'reg' };

/**
 * The kernel function of a lab file and nothing else: from the `__global__`
 * line (and the comment lines directly above it) to the closing brace in
 * column 0. Returns null when the file has no kernel, for example while it is
 * still loading.
 */
export function kernelOnly(src: string): { code: string; firstLine: number } | null {
  const lines = src.split('\n');
  const at = lines.findIndex((l) => /^\s*__global__\b/.test(l));
  if (at < 0) return null;
  let first = at;
  while (first > 0 && /^\s*\/\//.test(lines[first - 1])) first--;
  let last = at;
  while (last < lines.length - 1 && !/^}/.test(lines[last])) last++;
  return { code: lines.slice(first, last + 1).join('\n'), firstLine: first + 1 };
}

// The book's own worked case (ch. 1, p. 9) and the ceiling this page asks for, from the model chapter 1 runs on.
const BOOK_CASE = speedup(0.3, 100);
const BOOK_LIMIT = ceiling(0.3);
const DECODE_LIMIT = Math.round(ceiling(0.9));

// Prerequisite questions drawn from chapter 1 (from the notes for that chapter; it has no code).
export const questionsS20: Record<string, Question> = {
  'ch02/start/q-amdahl': {
    kind: 'numeric',
    id: 'ch02/start/q-amdahl',
    prompt: (
      <>
        In one decode step of a model, 90% of the time is spent in operations you could move into kernels. The other
        10% stays on the CPU: tokenising, the sampling logic, the loop itself. Suppose the kernel part became
        infinitely fast. What is the largest overall speed-up, as a factor?
      </>
    ),
    answer: DECODE_LIMIT,
    tol: 0.001,
    unit: '×',
    explain: (
      <>
        <p>
          The 10% is still there, so the run can shrink to one tenth of its time and no further: 1 / 0.1 ={' '}
          {DECODE_LIMIT}. The book makes the point with a harsher case: when 30% of a run is made 100 times faster,
          the whole run is only {BOOK_CASE.toFixed(2)} times faster, and with an infinitely fast 30% it is{' '}
          {BOOK_LIMIT.toFixed(2)}. The decode-step framing is ours; the arithmetic is the book's.
        </p>
        <p>
          Keep this ledger open for the whole chapter. The copies between host and device that you are about to
          write are not kernel work. They sit on the 10% side and make it larger, which is why the book warns that
          the program it builds here is inefficient as a pattern (
          <a href={href('ch02', '2.3')}>section 2.3</a>).
        </p>
      </>
    ),
    source: 'ch. 1 §1.3, p. 9',
  },
  'ch02/start/q-design': {
    kind: 'choice',
    id: 'ch02/start/q-design',
    prompt: (
      <>
        A CPU core is built to finish one thread's work as soon as possible. A GPU is built to finish the most work
        per second across very many threads, each of them slower. Which workload belongs on the GPU side?
      </>
    ),
    options: [
      {
        label: 'One long chain of steps in which every step needs the result of the one before.',
        why: 'There is never more than one thing ready to run, so only the speed of a single thread matters. That is what a CPU core is built for.',
      },
      {
        label: 'The same small computation applied to millions of elements, none of which needs another.',
        correct: true,
        why: 'Millions of pieces of work are ready at once, so throughput decides.',
      },
      {
        label: 'A handful of large tasks with heavy, unpredictable branching.',
        why: 'A few threads, and the branch prediction and large caches that serve them are what the CPU spent its area on.',
      },
      {
        label: 'Any of them. A GPU is the faster processor.',
        why: 'With one or a few threads a CPU wins. The book is explicit that applications should use both.',
      },
    ],
    explain: (
      <p>
        The second workload has a name, data parallelism, and it is the subject of{' '}
        <a href={href('ch02', '2.1')}>section 2.1</a>. The first is the case to learn to recognise early: a GPU has
        nothing to offer a chain. The generation loop of a language model is such a chain from one token to the
        next, which is why the kernels you write speed up the work inside a step and not the number of steps.
      </p>
    ),
    source: 'ch. 1 §1.1, pp. 3–6 (the generation-loop remark is ours)',
  },
  'ch02/start/q-challenge': {
    kind: 'choice',
    id: 'ch02/start/q-challenge',
    prompt: (
      <>
        Chapter 1 names four difficulties of parallel programming. A perfectly written vector addition,{' '}
        <code>C[i] = A[i] + B[i]</code> with one thread for each <code>i</code>, runs into one of them first. Which?
      </>
    ),
    options: [
      {
        label: 'Algorithm work: the parallel version performs more operations than the loop.',
        why: 'It performs exactly n additions, the same as the loop.',
      },
      {
        label: 'Memory-bound execution: the speed of moving bytes sets the pace, not the arithmetic.',
        correct: true,
      },
      {
        label: 'Input sensitivity: some threads get much more work than others.',
        why: 'Every element costs the same, whatever the values are.',
      },
      {
        label: 'Synchronisation overhead: threads wait for each other.',
        why: 'No thread needs anything from another thread. The book calls such work embarrassingly parallel.',
      },
    ],
    explain: (
      <p>
        Each thread does one addition and three memory accesses of 4 bytes each: two loads and a store. That is 12
        bytes moved for one operation, so the kernel can go only as fast as bytes travel between the GPU's memory and
        its arithmetic units. You will write this kernel in this chapter and measure it. The analysis that tells you
        how far from the limit it is comes in <a href={href('ch05', '5.1')}>chapter 5, section 5.1</a>.
      </p>
    ),
    source: 'ch. 1 §1.4, pp. 11–12; the 12 bytes per operation are from ch. 5 §5.1, p. 96',
  },
  'ch02/start/q-speedup': {
    kind: 'numeric',
    id: 'ch02/start/q-speedup',
    prompt: <>A job takes 200 s on system A and 10 s on system B. What is the speed-up of B over A?</>,
    answer: 20,
    unit: '×',
    explain: (
      <p>
        Time on the slower system divided by time on the faster one: 200 / 10 = 20. The book adds a caution that
        applies to every such figure. It also reflects how badly system A suited the job, so a large number can mean
        a good GPU kernel or a poor CPU baseline. <a href={href('ch02', 'labs')}>Lab 2.1</a> asks you to predict
        exactly such a ratio for your own machine before you measure it.
      </p>
    ),
    source: 'ch. 1 §1.3, pp. 9–10',
  },
};

export function S20() {
  const file = useLabFile(LAB1);
  const kernel = file ? kernelOnly(file) : null;
  return (
    <>
      <p>
        In PyTorch, <code>c = a + b</code> on two CUDA tensors is one line. Under it are four things you have never
        had to write: memory allocated on the device for <code>c</code>, possibly a copy of data across the boundary
        between the CPU's memory and the GPU's, a function whose whole body is one addition, and a call that starts
        one thread for every element. This chapter is those four things written by hand, for the simplest computation
        there is.
      </p>
      <Source pages="21" />
      <p>
        The book builds one complete program that adds two vectors. It splits the program between the CPU, called
        the host, and the GPU, called the device. It allocates device memory, copies the inputs in, writes the
        function every thread runs, launches the threads, and copies the result out. After it you can write and
        launch any one-dimensional elementwise kernel, size its launch for any length, and say exactly how many
        threads run and how many of them do useful work.
      </p>
      <Beyond>
        <p>
          The PyTorch connection is ours. Two parts of it were read in PyTorch's source: its CUDA allocator ends in{' '}
          <code>cudaMalloc</code>, under a caching layer so that most tensors never reach it, and{' '}
          <code>tensor.to("cuda")</code> ends in a memory copy of the host-to-device kind. Both calls are taught in{' '}
          <a href={href('ch02', '2.4')}>section 2.4</a>. Read on 2026-11-20.
        </p>
      </Beyond>

      <h2>The kernel you will be able to write</h2>
      <p>
        This is where the chapter ends up. Every thread runs this function once. It works out which element is its
        own, checks that the element exists, and adds one pair of numbers. Do not study it yet. Notice only how
        little there is, and that there is no loop.
      </p>
      <Source section="2.5" pages="35" figs="Fig. 2.10" />
      <Code
        tabs={[{ label: 'CUDA', code: kernel?.code ?? file ?? '// loading labs/' + LAB1, tiers: VEC_TIERS }]}
        firstLine={kernel?.firstLine ?? 1}
        caption={
          <>
            The kernel from the reference solution of lab 2.1, <code>labs/{LAB1}</code>, shown without the host code
            around it. It is the book's Fig. 2.10 with <code>const</code> added to the two arrays that are only read.
            The array names carry the colour of global memory and <code>i</code> the colour of a register.
          </>
        }
      />
      <Aside>
        <p>
          <b>Colour in this course</b> means where data lives. This chapter uses two of the four:{' '}
          <span className="chip reg">register</span> for a thread's private variables and{' '}
          <span className="chip global">global memory</span> for arrays on the device. Red always means something
          failed.
        </p>
      </Aside>

      <h2>The road through the chapter</h2>
      <ul>
        <li>
          <b>Data parallelism</b>: the property of a computation that makes one thread per output possible, and how
          to check for it (<a href={href('ch02', '2.1')}>section 2.1</a>).
        </li>
        <li>
          <b>Host and device</b>: one host thread that launches grids of device threads, and what has happened when
          a launch statement returns (<a href={href('ch02', '2.2')}>section 2.2</a>).
        </li>
        <li>
          <b>The host stub</b>: the three parts of the host function around a kernel, and why round trips between
          the two memories cost (<a href={href('ch02', '2.3')}>section 2.3</a>).
        </li>
        <li>
          <b>Device memory and transfers</b>: allocating, copying in each direction, freeing, checking errors (
          <a href={href('ch02', '2.4')}>section 2.4</a>).
        </li>
        <li>
          <b>The kernel and its index</b>: how a thread finds its element, and the test that stops surplus threads (
          <a href={href('ch02', '2.5')}>section 2.5</a>).
        </li>
        <li>
          <b>The launch</b>: the two numbers in the launch statement and how many blocks a given length needs (
          <a href={href('ch02', '2.6')}>section 2.6</a>).
        </li>
        <li>
          <b>What the compiler does</b> with a file that holds code for two processors (
          <a href={href('ch02', '2.7')}>section 2.7</a>).
        </li>
      </ul>

      <h2>The same program on AMD and Apple GPUs</h2>
      <p>
        On AMD's HIP the program keeps this structure exactly: the kernel is spelled the same, and the allocation
        and copy calls are the same calls with <code>hip</code> in place of <code>cuda</code>. On Apple silicon a
        Metal buffer in shared storage is one piece of memory that the CPU and the GPU both see, so the two copy
        steps have nothing to do and disappear.
      </p>
      <Beyond>
        <p>
          The book covers CUDA only. Each section of this chapter says where HIP and Metal differ, checked against
          AMD's and Apple's documentation on 2026-11-20.
        </p>
      </Beyond>

      <h2>What the labs ask of you</h2>
      <p>
        If you have done the labs of chapter 5, expect a difference. There the lab driver allocated the buffers and
        moved the data, and you wrote a kernel and its launch. Here the host side is yours too: allocation, both
        copies, the launch, the error checks and the free. There are <a href={href('ch02', 'labs')}>four labs</a>:
      </p>
      <ul>
        <li>2.1, vector addition from the first allocation to the last free;</li>
        <li>2.2, where the time goes between copies and kernel;</li>
        <li>
          2.3, milestone M0 of the course project: the elementwise operations a transformer needs (bias add,
          residual add, scale);
        </li>
        <li>2.4, a bug hunt in a host stub that almost works.</li>
      </ul>

      <h2>A check on chapter 1</h2>
      <p>
        Four questions on what this chapter assumes from chapter 1. That chapter has no code, so these are about
        reasoning with its numbers and its vocabulary. They are worth getting wrong now rather than later; each one
        comes back in <a href={href('review')}>review</a>.
      </p>
      <Ask q={questionsS20['ch02/start/q-amdahl']} label="From chapter 1" />
      <Ask q={questionsS20['ch02/start/q-design']} label="From chapter 1" />
      <Ask q={questionsS20['ch02/start/q-challenge']} label="From chapter 1" />
      <Ask q={questionsS20['ch02/start/q-speedup']} label="From chapter 1" />
    </>
  );
}
