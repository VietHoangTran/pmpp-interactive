import type { Chapter } from '../types';

// One file per section, each loaded on demand and exporting its component and its
// questions (chapter 5 is the template).
export const chapter: Chapter = {
  id: 'ch20',
  num: '20',
  title: 'Large language models',
  pages: '477–512',
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '477–478',
      lede: 'See which of your milestones this chapter composes into attention, and check the reductions, intensity arithmetic and layouts it assumes you remember.',
      load: () => import('./S200').then((m) => ({ Component: m.S200, questions: m.questionsS200 })),
    },
    {
      id: '20.1', num: '20.1', title: 'Transformer architecture', pages: '478–482',
      lede: 'Place every block of a decoder layer on a kernel you have written or will write, and say where its arithmetic goes as the context grows.',
      load: () => import('./S201').then((m) => ({ Component: m.S201, questions: m.questionsS201 })),
    },
    {
      id: '20.2', num: '20.2', title: 'Multi-head attention', pages: '482–486',
      lede: 'Write down the shape of every matrix in one causal attention head, say which entries the mask removes, and why subtracting the row max changes nothing but the risk of overflow.',
      load: () => import('./S202').then((m) => ({ Component: m.S202, questions: m.questionsS202 })),
    },
    {
      id: '20.3', num: '20.3', title: 'Implementing attention in CUDA', pages: '486–488',
      lede: 'Write a causal mask-and-softmax kernel with one block per row, say what each of its barriers protects, and count the global traffic the unfused pipeline pays for S and P.',
      load: () => import('./S203').then((m) => ({ Component: m.S203, questions: m.questionsS203 })),
    },
    {
      id: '20.4', num: '20.4', title: 'KV caching', pages: '488–492',
      lede: 'Predict exactly which entries change when one token is added, and explain why a decode step is a bandwidth-bound matrix-vector pass over the whole cache while prefill is a compute-bound GEMM.',
      load: () => import('./S204').then((m) => ({ Component: m.S204, questions: m.questionsS204 })),
    },
    {
      id: '20.5', num: '20.5', title: 'Flash attention', pages: '492–504',
      lede: 'Merge softmax results tile by tile with a running max and denominator, and write the single-kernel forward pass: who owns which data, which barriers it needs, what the book’s listing gets wrong, and what changes on HIP and Metal.',
      load: () => import('./S205').then((m) => ({ Component: m.S205, questions: m.questionsS205 })),
    },
    {
      id: '20.6', num: '20.6', title: 'KV cache arithmetic intensity and memory requirement', pages: '504–508',
      lede: 'Size a KV cache for any model, batch and precision against your device’s memory, and compute why batching lifts the projections’ intensity but not decode attention’s.',
      load: () => import('./S206').then((m) => ({ Component: m.S206, questions: m.questionsS206 })),
    },
    {
      id: '20.7', num: '20.7', title: 'Alleviating the memory requirements of the attention mechanism', pages: '508–510',
      lede: 'Compare MHA, GQA, MQA and MLA by cache size and decode intensity with the definitions straightened out, and explain how paging removes reserved but unused cache memory.',
      load: () => import('./S207').then((m) => ({ Component: m.S207, questions: m.questionsS207 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Five labs: build milestones M6 and M7 one kernel at a time, hunt the race the book’s flash kernel has, and measure decode against the intensity formula on your devices.',
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '20.8', title: 'Summary and exercises', pages: '510–511',
      lede: 'What to carry forward, a mixed check, where the book’s exercises and errata are worked, and milestones M6 and M7.',
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
