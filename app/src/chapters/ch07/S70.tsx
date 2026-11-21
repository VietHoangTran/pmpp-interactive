import type { ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source } from '../../components/notes';
import { href } from '../../lib/router';

// "Before you start" for chapter 7: what the chapter is for, the four kernels it builds,
// one honest bridge to LLM inference, and five prerequisite questions drawn from the
// notes for chapters 3 to 5.

export const questionsS70: Record<string, Question> = {
  'ch07/start/q-blur-corner': {
    kind: 'choice',
    id: 'ch07/start/q-blur-corner',
    prompt: (
      <>
        Chapter 3&rsquo;s blur kernel with <code>BLUR_SIZE = 1</code>, against a convolution with a 3×3 filter whose
        nine weights are all 1/9 and with zero ghost cells (missing neighbours count as 0). Do the two agree at the
        corner pixel (0, 0)?
      </>
    ),
    options: [
      {
        label: 'Yes: the blur also counts the five missing neighbours as zeros and divides its sum by 9.',
        why: 'The blur counts the pixels it actually added (4 at a corner) and divides by that count. They sum the same four pixels, but divide by different numbers.',
      },
      {
        label: 'No: the blur divides its sum by 4 and the convolution by 9, so the convolution is darker there.',
        correct: true,
      },
      {
        label: 'No: the convolution reads five pixels outside the image at the corner, so its sum is garbage.',
        why: 'Zero ghost cells are a defined value. Nothing outside the image is read; the missing neighbours are 0 by definition.',
      },
    ],
    explain: (
      <p>
        The blur kernel counts how many pixels it actually added (<code>pixels</code> = 4 at a corner, 6 on an edge)
        and divides by that. A convolution with fixed weights of 1/9 effectively adds five zeros and divides by 9, so
        it is 4/9 as bright at the corner. Neither is wrong: what a missing neighbour is worth is a policy you choose,
        and section 7.1 shows two more.
      </p>
    ),
    source: 'ch. 3 §3.3, pp. 55–59; ch. 7 §7.1, p. 162',
  },
  'ch07/start/q-rowmajor-patch': {
    kind: 'numeric',
    id: 'ch07/start/q-rowmajor-patch',
    prompt: (
      <>
        A 16-wide image is stored row-major in a flat array. The 3×3 patch centred on (row 3, column 5) starts its first
        row at which flat index?
      </>
    ),
    answer: 36,
    explain: (
      <p>
        The patch&rsquo;s upper-left corner is one row up and one column left of the centre: (row 2, column 4). Its
        flat index is row·width + column = 2·16 + 4 = 36. Section 7.2 computes exactly this corner for every thread, as{' '}
        <code>(outRow - r, outCol - r)</code>.
      </p>
    ),
    source: 'ch. 3 §3.2, pp. 48–55',
  },
  'ch07/start/q-divergence': {
    kind: 'numeric',
    id: 'ch07/start/q-divergence',
    prompt: (
      <>
        A 1D kernel with 256 threads per block runs over <code>n = 1000</code> elements with a grid of 4 blocks, and
        warps are 32 threads wide. How many warps diverge at <code>if (i &lt; n)</code>?
      </>
    ),
    answer: 1,
    unit: 'warps',
    explain: (
      <p>
        The grid has 1024 threads in 32 warps. Only the warp holding elements 992 to 1023 has both threads that pass
        (992 to 999) and threads that do not (1000 to 1023). Every warp before it passes entirely. Section 7.2 asks the
        same question of the convolution kernel, whose edge threads skip different numbers of filter taps.
      </p>
    ),
    source: 'ch. 4 §4.5, pp. 79–82',
  },
  'ch07/start/q-shared-flopb': {
    kind: 'numeric',
    id: 'ch07/start/q-shared-flopb',
    prompt: (
      <>
        An inner loop computes <code>acc += w_s[k] * x[i + k]</code>, where <code>w_s</code> is a{' '}
        <code>float</code> array in shared memory and <code>x</code> a <code>float</code> array in global memory. What
        is its ratio of floating-point operations to bytes read from global memory, counting only the loop?
      </>
    ),
    answer: 0.5,
    tol: 0.001,
    unit: 'FLOP/B',
    explain: (
      <p>
        Each iteration does two operations (a multiply and an add) and reads one 4-byte <code>float</code> from global
        memory. The shared-memory read costs no global bandwidth. 2 / 4 = 0.5. With both operands in global memory it
        would be 2 / 8 = 0.25, which is exactly where this chapter&rsquo;s first kernel starts.
      </p>
    ),
    source: 'ch. 5 §5.1–5.2, pp. 93–104',
  },
  'ch07/start/q-constant-copies': {
    kind: 'choice',
    id: 'ch07/start/q-constant-copies',
    prompt: (
      <>
        A kernel uses <code>__constant__ float table[64]</code> declared at file scope,{' '}
        <code>__shared__ float tile[256]</code>, and a local <code>float acc</code>. It is launched on one device with
        1000 blocks of 256 threads. How many copies of <code>table</code> exist while it runs?
      </>
    ),
    options: [
      { label: 'One.', correct: true },
      { label: '1000, one per block.', why: 'That is tile: shared memory has block scope.' },
      { label: '256,000, one per thread.', why: 'That is acc: a local scalar lives in a register of each thread.' },
      { label: 'One per streaming multiprocessor.', why: 'Caches may hold copies of its lines, but the variable itself exists once per device.' },
    ],
    explain: (
      <p>
        A <code>__constant__</code> variable has grid scope and lives for the whole application; CUDA&rsquo;s guide
        describes it as having &ldquo;a distinct object per device&rdquo;. <code>tile</code> has one copy per block (1000
        in all) and <code>acc</code> one per thread. Section 7.4 puts the convolution filter in exactly such a variable.
      </p>
    ),
    source: 'ch. 5 §5.2, pp. 98–104; notes/today/ch07-facts.md A1',
  },
};

const MAP: { kernel: string; fig: string; fixes: string; flopb: ReactNode; section: string }[] = [
  {
    kernel: 'Basic: one thread per output',
    fig: 'Fig. 7.7, p. 166',
    fixes: 'The starting point: correct at the edges, nothing reused on purpose.',
    flopb: <>0.25, whatever the filter size (p. 167)</>,
    section: '7.2',
  },
  {
    kernel: 'Filter in constant memory',
    fig: 'Fig. 7.9, p. 170',
    fixes: 'Filter reads stop costing DRAM bandwidth.',
    flopb: <>0.5 (p. 171)</>,
    section: '7.4',
  },
  {
    kernel: 'Tiled, halo cells in shared memory',
    fig: 'Fig. 7.12, p. 174',
    fixes: 'Each input element is fetched once per block, not once per tap.',
    flopb: (
      <>
        A formula in filter width and tile width that approaches the ideal m²/4; 5.42 for a 5×5 filter and 28×28 output
        tiles (pp. 176–177)
      </>
    ),
    section: '7.5',
  },
  {
    kernel: 'Tiled, halo cells left to the cache',
    fig: 'Fig. 7.15, p. 178',
    fixes: 'Input and output tiles the same size, a power of two.',
    flopb: <>The book gives no figure</>,
    section: '7.6',
  },
];

export function S70() {
  return (
    <>
      <p>
        Convolution is the first of the book&rsquo;s parallel patterns. It runs through signal, image and video
        processing and computer vision, usually as a filter; chapter 3&rsquo;s blur kernel was one.
      </p>
      <Aside>
        <p>
          The chapter opens by saying Gaussian filters can be used to sharpen edges (p. 159). In the tools you use a
          Gaussian filter is a blur: torchvision&rsquo;s <code>GaussianBlur</code> &ldquo;blurs image with randomly
          chosen Gaussian blur&rdquo; (facts D5). This course calls it a smoothing filter.
        </p>
      </Aside>
      <p>
        It is a useful pattern to start with because it is easy and awkward at the same time. Every output is
        independent of every other, so there is parallelism to spare. But neighbouring outputs read almost the same
        inputs, and the outputs near the edges need inputs that do not exist. Handling both well is what the chapter is
        about: tiling, and staging inputs on the chip.
      </p>
      <Source pages="159" />
      <p>
        You will not write many image filters on the way to fast LLM inference. You will use the two techniques the
        chapter teaches through them:
      </p>
      <ul>
        <li>
          <b>constant memory</b> for a small read-only table that every thread reads in the same order;
        </li>
        <li>
          <b>tiling when the input tile is larger than the output tile</b>: a block needs a border of extra inputs, the{' '}
          <b>halo</b>, around the outputs it owns.
        </li>
      </ul>
      <p>
        And you will practise the method from chapter 5 on a new shape: each kernel is judged by its floating-point
        operations per byte of global memory traffic, now as a function of the filter size and the tile size.
      </p>

      <h2>The four kernels</h2>
      <p>
        The chapter builds one kernel per step. Each row says what the step fixes and what the book computes for it.
        The numbers are the book&rsquo;s model, not measurements; the labs measure.
      </p>
      <table className="data wide" data-map style={{ maxWidth: '58rem' }}>
        <thead>
          <tr>
            <th>Kernel</th>
            <th>In the book</th>
            <th>What it fixes</th>
            <th>FLOP/B, as the book computes it</th>
          </tr>
        </thead>
        <tbody>
          {MAP.map((m) => (
            <tr key={m.section} data-section={m.section}>
              <td>
                <a href={href('ch07', m.section)}>{m.kernel}</a> <span className="muted">(§{m.section})</span>
              </td>
              <td>{m.fig}</td>
              <td>{m.fixes}</td>
              <td>{m.flopb}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>
        Section <a href={href('ch07', '7.1')}>7.1</a> defines what all four compute, and section{' '}
        <a href={href('ch07', '7.3')}>7.3</a> explains why the first one is so far from the ideal. Two labs on the{' '}
        <a href={href('ch07', 'labs')}>labs page</a> put the filter in constant memory three ways and build the tiled
        kernel with halos, each with a prediction to commit before you measure.
      </p>

      <h2>Where this sits under the frameworks you use</h2>
      <p>
        What PyTorch calls a convolution layer computes this chapter&rsquo;s formula. <code>torch.nn.Conv2d</code> is
        documented as the &ldquo;valid 2D cross-correlation operator&rdquo;, which is the book&rsquo;s formula: it does
        not flip the filter (section 7.1 shows what that means). With <code>padding=r</code> and zeros as padding it is
        the book&rsquo;s kernel with zero ghost cells.
      </p>
      <Beyond>
        <p>
          This whole part is ours, checked on 2026-11-20 against PyTorch 2.15&rsquo;s documentation and the llama.cpp
          and MLX sources (notes/today/ch07-facts.md, D1 to D3).
        </p>
      </Beyond>
      <p>
        Production libraries rarely run it the way the book does. They mostly turn a convolution into a matrix
        multiplication, which is chapter 19 (<a href={href('ch19', '19.3')}>§19.3</a>): llama.cpp&rsquo;s <code>ggml_conv_2d</code> is <code>ggml_im2col</code>{' '}
        followed by <code>ggml_mul_mat</code>, and MLX chooses between Winograd, implicit GEMM, explicit GEMM and a
        depthwise kernel.
      </p>
      <p>
        In LLM inference its direct use is narrow: vision and audio front-ends, and the short causal 1D convolution of
        Mamba-style and hybrid models, which <a href={href('ch07', '7.5')}>section 7.5</a> follows into llama.cpp: its
        kernel for long prompts uses that section&rsquo;s technique.
      </p>
      <p>
        What carries over, and why the chapter is worth your time: read-only values that every thread reads the same
        way (sampling parameters, tensor shapes) belong in constant memory or in kernel arguments passed by value;
        halo tiling applies to any sliding-window operation; and the bookkeeping of arithmetic intensity against tile
        size is exactly how GEMM tiles are sized in chapter 15.
      </p>

      <h2>Before you start</h2>
      <p>
        Five questions on what this chapter assumes from chapters 3 to 5. Each one names the page it comes from, and
        each comes back in <a href={href('review')}>review</a>.
      </p>
      <Ask q={questionsS70['ch07/start/q-blur-corner']} label="From chapter 3" />
      <Ask q={questionsS70['ch07/start/q-rowmajor-patch']} label="From chapter 3" />
      <Ask q={questionsS70['ch07/start/q-divergence']} label="From chapter 4" />
      <Ask q={questionsS70['ch07/start/q-shared-flopb']} label="From chapter 5" />
      <Ask q={questionsS70['ch07/start/q-constant-copies']} label="From chapter 5" />
    </>
  );
}
