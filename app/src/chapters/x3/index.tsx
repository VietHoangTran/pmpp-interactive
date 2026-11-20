import type { Chapter } from '../types';

// Planned. The sections are added when the chapter is built (see PLAN.md).
export const chapter: Chapter = {
  id: 'x3',
  num: 'X3',
  title: 'Reading real backends',
  pages: 'not in the book',
  sections: [
    // TEMPORARY (group 1 builder): start, X3.1 and X3.2 wired for testing; the integrator owns this list.
    {
      id: 'start', num: '', title: 'Before you start', pages: null,
      lede: 'See what reading a real backend means in this module, which three projects and commits it reads, and check what it assumes from chapters 2 to 22 and module X1.',
      load: () => import('./SX30').then((m) => ({ Component: m.SX30, questions: m.questionsSX30 })),
    },
    {
      id: 'X3.1', num: 'X3.1', title: 'Finding the kernel for an op', pages: null,
      lede: 'Start from the one place each project turns an op into a backend call, follow one name, and land in the kernel on CUDA, HIP and Metal.',
      load: () => import('./SX31').then((m) => ({ Component: m.SX31, questions: m.questionsSX31 })),
    },
    {
      id: 'X3.2', num: 'X3.2', title: 'Elementwise and gather ops', pages: null,
      lede: 'Read the three projects’ elementwise and embedding kernels as chapter 2’s kernel with chapter 6’s coarsening and vector loads, and say how each one picks a variant on CUDA and on Metal.',
      load: () => import('./SX32').then((m) => ({ Component: m.SX32, questions: m.questionsSX32 })),
    },
    // TEMPORARY (group 2 builder): X3.3, X3.4 and X3.5 wired for testing; the integrator owns this list.
    {
      id: 'X3.3', num: 'X3.3', title: 'Row reductions: RMSNorm and softmax', pages: null,
      lede: 'Recognise chapter 10’s block reduction in the RMSNorm and softmax kernels of all three projects, see how an online softmax reads the row twice where lab 10.3’s reads it three times, and say which exponential each kernel calls.',
      load: () => import('./SX33').then((m) => ({ Component: m.SX33, questions: m.questionsSX33 })),
    },
    {
      id: 'X3.4', num: 'X3.4', title: 'Matrix products: the shape picks the kernel', pages: null,
      lede: 'Predict which kernel each project runs for a matrix product of a given shape, from one token to a full batch, and read MLX’s GEMV as lab 22.1’s kernel.',
      load: () => import('./SX34').then((m) => ({ Component: m.SX34, questions: m.questionsSX34 })),
    },
    {
      id: 'X3.5', num: 'X3.5', title: 'Attention: own kernels, libraries and submodules', pages: null,
      lede: 'Say which attention kernel each project runs for prefill and for decode, find it even when it is not in the project’s tree, and read MLX’s CUDA decode kernel against lab 20.4.',
      load: () => import('./SX35').then((m) => ({ Component: m.SX35, questions: m.questionsSX35 })),
    },
    // TEMPORARY (group 3 builder): X3.6 and X3.7 wired for testing; the integrator owns this list.
    {
      id: 'X3.6', num: 'X3.6', title: 'Sampling and greedy decoding', pages: null,
      lede: 'Find each project’s sampler and argmax, say where each runs by default, and predict which token each one picks on an exact tie.',
      load: () => import('./SX36').then((m) => ({ Component: m.SX36, questions: m.questionsSX36 })),
    },
    {
      id: 'X3.7', num: 'X3.7', title: 'The Llama step: RMSNorm, RoPE and SwiGLU', pages: null,
      lede: 'Lay out one Llama decode step as launches of the course’s kernels, add the three pieces the course has not built, and keep the two RoPE pairings apart.',
      load: () => import('./SX37').then((m) => ({ Component: m.SX37, questions: m.questionsSX37 })),
    },
    // TEMPORARY (group 4 builder): X3.8 and X3.9 wired for testing; the integrator owns this list.
    {
      id: 'X3.8', num: 'X3.8', title: 'Real checkpoints: formats and the traps between them', pages: null,
      lede: 'Read a llama2.c .bin header and a safetensors header by hand, check a download by its size, and list the conversions a Hugging Face Llama needs before the course’s kernels can run it.',
      load: () => import('./SX38').then((m) => ({ Component: m.SX38, questions: m.questionsSX38 })),
    },
    {
      id: 'X3.9', num: 'X3.9', title: 'Testing against a reference, and contributing', pages: null,
      lede: 'Compare a GPU run with a reference the way the projects do, explain why exact greedy agreement can fail late without any kernel being wrong, run your own kernel inside MLX or PyTorch, and know what each project asks of a kernel change.',
      load: () => import('./SX39').then((m) => ({ Component: m.SX39, questions: m.questionsSX39 })),
    },
  ],
};
