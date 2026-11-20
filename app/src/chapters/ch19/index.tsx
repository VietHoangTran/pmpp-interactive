import type { Chapter } from '../types';

// Chapter 19 is standard depth: a page per book section, one lab, and a summary. Each
// section loads on demand and exports its component and its questions.
export const chapter: Chapter = {
  id: 'ch19',
  num: '19',
  title: 'Convolutional neural networks',
  pages: '453–475',
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '453–454',
      lede: 'See what a convolution chapter is doing in a course about LLM kernels, what it builds, and the one idea from it that carries over, then check what it assumes from chapters 3, 5 and 7.',
      load: () => import('./S190').then((m) => ({ Component: m.S190, questions: m.questionsS190 })),
    },
    {
      id: '19.1', num: '19.1', title: 'Convolutional neural networks', pages: '454–459',
      lede: 'Compute any output of a convolutional layer by hand from its input maps and filter bank, and give the shapes of X, F and Y for any layer, batched or not.',
      load: () => import('./S191').then((m) => ({ Component: m.S191, questions: m.questionsS191 })),
    },
    {
      id: '19.2', num: '19.2', title: 'A CUDA convolutional layer kernel', pages: '460–463',
      lede: 'Map a 3D grid onto samples, output maps and tiles, write the direct kernel with the bounds checks the book leaves out, and explain why it cannot reuse an input tile across output maps.',
      load: () => import('./S192').then((m) => ({ Component: m.S192, questions: m.questionsS192 })),
    },
    {
      id: '19.3', num: '19.3', title: 'Formulating a convolutional layer as GEMM', pages: '463–472',
      lede: 'Build the unrolled input matrix for any layer, say what it costs, trace any of its elements back to the input with Eq. 19.5, and read the tiled kernel that never stores it.',
      load: () => import('./S193').then((m) => ({ Component: m.S193, questions: m.questionsS193 })),
    },
    {
      id: '19.4', num: '19.4', title: 'The cuDNN library', pages: '472–474',
      lede: 'Describe a convolution to a vendor library the way cuDNN, MIOpen and Apple’s frameworks expect, translate between their names and the chapter’s, and say what the library decides for you.',
      load: () => import('./S194').then((m) => ({ Component: m.S194, questions: m.questionsS194 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'One lab on your own GPUs: a convolutional layer computed directly and as unroll plus the project’s GEMM, timed on early, late and pointwise layer shapes against your prediction.',
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '19.5', title: 'Summary and exercises', pages: '474–475',
      lede: 'What to carry forward, a mixed check, where each of the book’s exercises is worked, the PDF’s apparent errata for this chapter, and how the project’s GEMM was reused.',
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
