import type { Chapter } from '../types';

// Module X2 is not in the book: every section is ours, so no section carries book pages.
// Built in the shape of chapter 5: one file per section, each loaded on demand and
// exporting its component and its questions.
export const chapter: Chapter = {
  id: 'x2',
  num: 'X2',
  title: 'Quantised kernels',
  pages: 'not in the book',
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: null,
      lede: 'See which kernel runs under a quantised linear layer in torchao, MLX and llama.cpp, what this module builds on your GPUs, and check what it assumes from chapters 5, 6, 10, 20, 22 and appendix A.',
      load: () => import('./SX20').then((m) => ({ Component: m.SX20, questions: m.questionsSX20 })),
    },
    {
      id: 'X2.1', num: 'X2.1', title: 'Bytes per weight set the speed of decode', pages: null,
      lede: 'Predict a decode GEMV’s speed-up from the bytes each weight format moves, see why a separate dequantise step makes it slower than float32, and turn your own measurement into a fraction of your bandwidth.',
      load: () => import('./SX21').then((m) => ({ Component: m.SX21, questions: m.questionsSX21 })),
    },
    {
      id: 'X2.2', num: 'X2.2', title: 'Scales, zero points and blocks', pages: null,
      lede: 'Quantise a block of weights by the rules llama.cpp and MLX really use, predict which values clip, and find how block size, bit width and outliers trade error against bytes.',
      load: () => import('./SX22').then((m) => ({ Component: m.SX22, questions: m.questionsSX22 })),
    },
    {
      id: 'X2.3', num: 'X2.3', title: 'Two codes per byte', pages: null,
      lede: 'Pack and unpack 4-bit codes in the orders llama.cpp and MLX really use, read an 18-byte block without a misaligned load, and say what each backend gives you for 4-bit data.',
      load: () => import('./SX23').then((m) => ({ Component: m.SX23, questions: m.questionsSX23 })),
    },
    {
      id: 'X2.4', num: 'X2.4', title: 'A quantised GEMV, warp by warp', pages: null,
      lede: 'Map warps, lanes and loads onto rows of quantised blocks, watch which mappings coalesce and which lanes sit idle at the end of a row, and read the kernel lab X2.2 asks for.',
      load: () => import('./SX24').then((m) => ({ Component: m.SX24, questions: m.questionsSX24 })),
    },
    {
      id: 'X2.5', num: 'X2.5', title: 'Dequantise, or multiply in integers', pages: null,
      lede: 'Follow one block’s dot product through the float path real Metal kernels use and the integer path llama.cpp uses on CUDA and HIP, prove the offset folds out of the loop, and say what each backend offers for a byte dot product.',
      load: () => import('./SX25').then((m) => ({ Component: m.SX25, questions: m.questionsSX25 })),
    },
    {
      id: 'X2.6', num: 'X2.6', title: 'Where the accumulation must stay wide', pages: null,
      lede: 'Bound every partial sum of a quantised dot product, choose the accumulator for each level, and watch the narrow choices wrap, overflow or drift.',
      load: () => import('./SX26').then((m) => ({ Component: m.SX26, questions: m.questionsSX26 })),
    },
    {
      id: 'X2.7', num: 'X2.7', title: 'The quantised decode step', pages: null,
      lede: 'Put quantised linear layers into milestone M8’s decode step, predict the speed-up from the bytes of the whole step, and judge the error by logits and greedy tokens instead of by weights.',
      load: () => import('./SX27').then((m) => ({ Component: m.SX27, questions: m.questionsSX27 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Four labs on your own GPUs: an int8-weight GEMV against float32, a 4-bit GEMV on llama.cpp’s Q4_0 blocks, a hunt for four defects on the integer path, and milestone M9, the quantised decode step.',
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '', title: 'Summary', pages: null,
      lede: 'What to carry forward, a mixed check, where each task of the module is worked, where the sources disagree, and milestone M9.',
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
