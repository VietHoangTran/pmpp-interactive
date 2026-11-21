import { Ask, type Question } from '../../components/Ask';
import { Aside, Beyond, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { basicIntensity } from '../../sim/convai';
import { offset, phases, printedPhases } from '../../sim/convlib';

// Prerequisite questions drawn from the notes of chapters 3, 5, 7 and 15. Each explanation
// names where the idea was taught and the section of this chapter that builds on it. The
// numeric answers come from the tested models (convlib, convai), not from typed numbers.
const LENET_INNER = 25; // LeNet C1: C·K·K = 1·5·5 (notes, §19.3 derived)
const TILE = 16;

export const questionsS190: Record<string, Question> = {
  'ch19/start/q-index4d': {
    kind: 'numeric',
    id: 'ch19/start/q-index4d',
    prompt: (
      <>
        <code>X</code> holds a batch of images as an <code>N × C × H × W</code> array, stored row-major in a flat{' '}
        <code>float*</code>, with <code>C = 3</code> and <code>H = W = 32</code>. What is the flat index of{' '}
        <code>X[1][2][5][7]</code>?
      </>
    ),
    answer: offset('NCHW', { C: 3, H: 32, W: 32 }, 1, 2, 5, 7),
    explain: (
      <p>
        Peel off one dimension at a time: <code>((n·C + c)·H + h)·W + w = ((1·3 + 2)·32 + 5)·32 + 7 = 5287</code>. The
        chapter writes <code>X[n, c, h + p, w + q]</code> as pseudo-code and leaves this translation to you (section{' '}
        <a href={href('ch19', '19.2')}>19.2</a>), and line 28 of Fig. 19.11 is this very expression with the unfolding
        map inside it (section <a href={href('ch19', '19.3')}>19.3</a>). Taught in{' '}
        <a href={href('ch03', '3.2')}>ch. 3 §3.2</a>.
      </p>
    ),
    source: 'ch. 3 §3.2, p. 55 (the 3D form), and §3.4; ch. 19 pp. 461, 463, 471',
  },
  'ch19/start/q-reuse': {
    kind: 'choice',
    id: 'ch19/start/q-reuse',
    prompt: (
      <>
        A tiled GEMM with <code>T × T</code> tiles computes a <code>j × l</code> output <code>Y = A · B</code>. Ignoring
        caches, how many times is each element of <code>A</code> fetched from global memory?
      </>
    ),
    options: [
      { label: <>ceil(l / T): once by each block in its row of blocks.</>, correct: true },
      {
        label: 'Once: after the first block loads it, shared memory serves every other block.',
        why: 'That is the ideal. Every block that needs the element loads it into its own shared memory, and blocks do not share.',
      },
      {
        label: <>ceil(j / T): once by each block in its column of blocks.</>,
        why: 'That counts the blocks in one column of the output. An element of A is needed along its row of Y, which crosses the columns.',
      },
      {
        label: 'T times: once in each of the T phases of its block.',
        why: 'A block loads a given element of A in one phase only. Tiling divides the fetches by T compared with one thread per output; what is left is one fetch per block that needs the element.',
      },
    ],
    explain: (
      <p>
        An element in row r of <code>A</code> feeds every output in row r of <code>Y</code>, and those outputs are spread
        over ceil(l / T) blocks. Each of them fetches it once. For LeNet&rsquo;s first layer as a GEMM, a 6 × 784 output
        with T = 16, every filter weight is fetched {phases(784, TILE)} times. Larger tiles and larger matrices pay
        better, which is the argument section <a href={href('ch19', '19.3')}>19.3</a> ends on and section{' '}
        <a href={href('ch15', '15.2')}>15.2</a> pushes further. Taught in <a href={href('ch05', '5.3')}>ch. 5 §5.3</a>{' '}
        and <a href={href('ch15', '15.2')}>ch. 15 §15.2</a>.
      </p>
    ),
    source: 'ch. 5 §5.3, pp. 104–107; ch. 15 §15.2, pp. 351–352; ch. 19 p. 472',
  },
  'ch19/start/q-phases': {
    kind: 'numeric',
    id: 'ch19/start/q-phases',
    prompt: (
      <>
        A tiled GEMM with <code>T = 16</code>, written with the boundary checks of section 5.5, multiplies a 6 × 25
        matrix by a 25 × 784 matrix. How many times does its phase loop run?
      </>
    ),
    answer: phases(LENET_INNER, TILE),
    unit: 'phases',
    explain: (
      <p>
        ceil(25 / 16) = 2. In the second phase only 9 of the 16 columns of the left tile, and 9 of the 16 rows of the
        right one, exist; the checks fill the other 7 slots with zeros, which add nothing. These two matrices are
        LeNet&rsquo;s first layer, the filter bank <code>F</code> and the unfolded input <code>B</code>. Fig. 19.11, as
        printed, runs <code>(C*K*K)/TILE_WIDTH</code> phases, which truncates to {printedPhases(LENET_INNER, TILE)} here
        and silently drops {LENET_INNER - printedPhases(LENET_INNER, TILE) * TILE} of the 25 terms (section{' '}
        <a href={href('ch19', '19.3')}>19.3</a>). Taught in <a href={href('ch05', '5.5')}>ch. 5 §5.5</a>.
      </p>
    ),
    source: 'ch. 5 §5.5, pp. 112–115 (Fig. 5.13); ch. 19 p. 471',
  },
  'ch19/start/q-convratio': {
    kind: 'numeric',
    id: 'ch19/start/q-convratio',
    prompt: (
      <>
        Chapter 7&rsquo;s basic 2D convolution kernel gives each thread one output and reads the filter from global
        memory. With a 5 × 5 filter in single precision, how many FLOP does its inner loop do per byte it requests from
        global memory?
      </>
    ),
    answer: basicIntensity(4),
    tol: 0.01,
    unit: 'FLOP/B',
    explain: (
      <p>
        Each tap loads one input and one filter value, 8 bytes, for one multiply and one add: 2 / 8 = 0.25. The filter
        size changes how many taps there are, not what each one costs. The direct kernel of section{' '}
        <a href={href('ch19', '19.2')}>19.2</a> does the same per tap, over every input channel, so it inherits 0.25, and
        the book calls it limited by global-memory bandwidth. Taught in <a href={href('ch07', '7.3')}>ch. 7 §7.3</a>.
      </p>
    ),
    source: 'ch. 7 §7.3, p. 167; ch. 19 p. 463',
  },
  'ch19/start/q-xcorr': {
    kind: 'choice',
    id: 'ch19/start/q-xcorr',
    prompt: (
      <>
        Does PyTorch&rsquo;s <code>nn.Conv2d</code> flip the filter before sliding it over the input? Does chapter
        7&rsquo;s convolution formula?
      </>
    ),
    options: [
      {
        label: 'Neither does: both compute what frameworks call cross-correlation.',
        correct: true,
      },
      {
        label: 'PyTorch flips it, as signal processing does; the book’s formula does not.',
        why: 'PyTorch documents nn.Conv2d as a valid 2D cross-correlation, which has no flip.',
      },
      {
        label: 'The book flips it, as signal processing does; PyTorch does not.',
        why: 'Chapter 7’s formula walks the filter and the input in the same direction, so nothing is reversed.',
      },
      {
        label: 'Both flip it when the filter size is odd.',
        why: 'Flipping does not depend on the size of the filter, and neither of them flips.',
      },
    ],
    explain: (
      <p>
        A convolution in the strict sense reverses the filter; cross-correlation does not. Chapter 7&rsquo;s formula, and
        this chapter&rsquo;s layer (<code>X[c, h + p, w + q] * F[m, c, p, q]</code> in Fig. 19.3), read filter and input
        in the same direction. PyTorch documents <code>nn.Conv2d</code> as cross-correlation and calls cuDNN and MIOpen in
        that mode, so the book&rsquo;s weights and PyTorch&rsquo;s line up with no reversal. Section{' '}
        <a href={href('ch19', '19.4')}>19.4</a> comes back to which libraries document this and which do not. Taught in{' '}
        <a href={href('ch07', '7.1')}>ch. 7 §7.1</a>.
      </p>
    ),
    source: 'ch. 7 §7.1, pp. 159–164 (the no-flip reading is the notes’); ch. 19 p. 457; ch07-facts D1',
  },
};

// ---------------------------------------------------------------- the chapter at a glance
const MAP: [string, string, string][] = [
  ['19.1', 'Convolutional neural networks', 'Read a layer’s shapes: the input X, the filter bank F, the output Y, and what a batch adds'],
  ['19.2', 'A CUDA convolutional layer kernel', 'Give each thread one output, place the grid, and see why the kernel waits on memory'],
  ['19.3', 'Formulating a convolutional layer as GEMM', 'Unfold the input into a matrix, first explicitly, then tile by tile inside a GEMM with the index map of Eq. 19.5'],
  ['19.4', 'The cuDNN library', 'Describe a layer to cuDNN, MIOpen or Apple’s frameworks, and say what each one decides for you'],
];

export function S190() {
  const q = questionsS190;
  return (
    <>
      <p>
        This course is about the kernels behind LLM inference, and a decoder-only transformer block contains no
        convolution. Convolutions run before the language model sees anything, in the encoders that turn an image or a
        sound into tokens for a multimodal model. A vision encoder&rsquo;s patch embedding is a convolution whose stride
        equals its filter size, so its windows never overlap. A Whisper-style audio encoder opens with 1D convolutions.
        Some hybrid models also run a short causal 1D convolution inside their blocks, which you met in section{' '}
        <a href={href('ch07', '7.5')}>7.5</a>. You will meet convolutions, then, but they are not why this chapter is
        here.
      </p>
      <Beyond>
        <p>
          Where convolutions appear in LLM serving was read from llama.cpp&rsquo;s sources on 2026-11-20 (ch19-facts D2;
          ch07-facts D3).
        </p>
      </Beyond>
      <p>
        The reason is one move the chapter makes on p. 464, which it credits to Chellapilla and colleagues: rewrite the
        operator as a single matrix multiplication, so that one heavily tuned GEMM routine serves it. Every output of a
        convolutional layer is a dot product of some inputs with some weights, and a matrix product is a table of dot
        products. Lay the filters out as the rows of one matrix and the input patches as the columns of another, and
        the layer is a GEMM. Chapter 15 says attention layers are formulated as large matrix multiplications too. When
        an operator you need is a pile of dot products, look for the view in which it is one GEMM, and spend your effort
        on that one kernel.
      </p>
      <Source section="19.3" pages="463–464" />

      <h2>What the book sets out to do</h2>
      <p>
        The book&rsquo;s framing is historical. Deep learning, machine learning with deep hierarchies of feature
        extractors and very many parameters, waited a long time for two things: large datasets, and the arithmetic of
        GPUs. The book dates its take-off in industry to 2012. Convolutional layers suit GPUs, it says, because they do
        a lot of arithmetic for every byte they touch and offer plenty of independent work.
      </p>
      <Source pages="453–454" />
      <p>
        The plan it states is a basic CUDA kernel for the forward path of one convolutional layer, &ldquo;a few
        optimizations&rdquo;, and then the layer as a matrix multiplication. Read the middle item with care. The direct
        kernel gets no optimisation of its own: section 19.2 shows it, says it uses too much global-memory bandwidth,
        and leaves shared-memory tiling to you. The one optimisation the chapter develops is the GEMM formulation.
      </p>
      <Aside>
        <p>
          The intro promises optimisations on p. 454; tiling the direct kernel is left as an exercise on p. 463 (notes,
          &ldquo;Unclear in the PDF&rdquo;).
        </p>
      </Aside>

      <h2>The chapter at a glance</h2>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-map>
          <thead>
            <tr>
              <th>Section</th>
              <th>What you will be able to do</th>
            </tr>
          </thead>
          <tbody>
            {MAP.map(([id, title, what]) => (
              <tr key={id}>
                <td>
                  <a href={href('ch19', id)}>
                    {id} {title}
                  </a>
                </td>
                <td>{what}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p data-lab>
        One lab, on the <a href={href('ch19', 'labs')}>labs page</a>. Lab 19.1 builds the layer two ways, as the direct
        kernel of section 19.2 and as an explicit unfold followed by the project&rsquo;s M2 GEMM, lab 6.4&rsquo;s{' '}
        <code>launchLinear</code>, and times both on your device. You predict the outcome before you run it.
      </p>

      <h2>Underneath torch.nn.functional.conv2d</h2>
      <p>
        PyTorch keeps a list of backends for a convolution. Among them are cuDNN, MIOpen and MPS (an MPSGraph
        convolution on Apple silicon), the vendor libraries of section 19.4. There is also a slow CUDA path of its own,
        and that one is section 19.3 nearly line for line: for each sample it unfolds the input into a column matrix,
        then calls one GEMM with the weights. The weights need no copy, because <code>nn.Conv2d</code> stores them
        output channels first, the chapter&rsquo;s <code>M × C × K × K</code>.
      </p>
      <Beyond>
        <p>
          Read from PyTorch&rsquo;s <code>ConvUtils.h</code>, <code>ConvolutionMM2d.cu</code> and the 2.15 docs on
          2026-11-20 (ch19-facts D4).
        </p>
      </Beyond>
      <p>
        llama.cpp lowers convolutions the same way in its generic graph, an explicit unfold followed by a matrix
        multiplication, and its CUDA backend has a kernel that is Fig. 19.11 in production. Section 19.4 opens both.
      </p>
      <Today checked="2026-11-20 (ch19-facts D1, D2)">
        <p>
          llama.cpp&rsquo;s multimodal tools build a vision model&rsquo;s patch embedding with <code>ggml_conv_2d</code>,
          stride equal to the patch size and no padding, and its Whisper encoder runs <code>conv1d</code> layers.
        </p>
      </Today>
      <p>
        The chapter&rsquo;s kernels use only threads, blocks, shared memory and barriers, and for those one{' '}
        <code>.cu</code> file is set up to build for HIP unchanged through the course&rsquo;s{' '}
        <code>pmpp/vendors/hip.h</code> (documented, ch19-facts B1); it has not been compiled or run on AMD hardware here.
        That would stop being true if they used warp <code>_sync</code> intrinsics, cooperative groups, half-precision
        headers, CUB functors, matrix-core intrinsics or <code>__dp4a</code> (HIP has none), which they do not. On Metal they are rewritten in MSL with{' '}
        <code>threadgroup</code> arrays and <code>threadgroup_barrier</code>.
      </p>
      <Beyond>
        <p>The book covers CUDA only. Checked against vendor documentation (ch19-facts B1, C1).</p>
      </Beyond>

      <h2>Before you start</h2>
      <p>
        Five questions on what this chapter assumes from chapters 3, 5, 7 and 15. Each explanation points to where the
        idea was taught and to the section here that uses it. They come back in <a href={href('review')}>review</a>.
      </p>
      <Ask q={q['ch19/start/q-index4d']} label="From chapter 3" />
      <Ask q={q['ch19/start/q-reuse']} label="From chapters 5 and 15" />
      <Ask q={q['ch19/start/q-phases']} label="From chapter 5" />
      <Ask q={q['ch19/start/q-convratio']} label="From chapter 7" />
      <Ask q={q['ch19/start/q-xcorr']} label="From chapter 7" />
    </>
  );
}
