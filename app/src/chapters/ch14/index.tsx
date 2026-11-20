import type { Chapter } from '../types';

// Built on chapter 5's template: one file per section, each loaded on demand and exporting
// its component and its questions.
export const chapter: Chapter = {
  id: 'ch14',
  num: '14',
  title: 'Sorting',
  pages: '329–348',
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '329',
      lede: 'See where sorting sits under a top-p sampler, why top-k needs no full sort, and check what this chapter assumes from chapters 4, 6 and 10 to 13.',
      load: () => import('./S140').then((m) => ({ Component: m.S140, questions: m.questionsS140 })),
    },
    {
      id: '14.1', num: '14.1', title: 'Background', pages: '330–331',
      lede: 'Say what a sort must guarantee, decide when you need stability, and build a row-wise sort from two stable passes.',
      load: () => import('./S141').then((m) => ({ Component: m.S141, questions: m.questionsS141 })),
    },
    {
      id: '14.2', num: '14.2', title: 'Parallel odd-even sort', pages: '331–333',
      lede: 'Step odd-even sort across its threads, say why its pairs never collide, stop it correctly, and set its flag without a data race.',
      load: () => import('./S142').then((m) => ({ Component: m.S142, questions: m.questionsS142 })),
    },
    {
      id: '14.3', num: '14.3', title: 'Parallel merge sort', pages: '333–334',
      lede: 'Follow a merge tree stage by stage and say how a grid\'s blocks should be shared among its merges.',
      load: () => import('./S143').then((m) => ({ Component: m.S143, questions: m.questionsS143 })),
    },
    {
      id: '14.4', num: '14.4', title: 'Radix sort', pages: '334–336',
      lede: 'Run an LSD radix sort pass by pass, say why each pass must be stable, and turn float logits into keys that radix sort orders correctly, descending.',
      load: () => import('./S144').then((m) => ({ Component: m.S144, questions: m.questionsS144 })),
    },
    {
      id: '14.5', num: '14.5', title: 'Parallel radix sort', pages: '336–339',
      lede: 'Compute every key\'s destination in a radix pass from one exclusive scan, and split the book\'s one-kernel pass into kernels that run on every backend.',
      load: () => import('./S145').then((m) => ({ Component: m.S145, questions: m.questionsS145 })),
    },
    {
      id: '14.6', num: '14.6', title: 'Optimizing for memory coalescing', pages: '339–342',
      lede: 'Count the memory runs a radix scatter makes, and remove most of them with a local sort in shared memory and a scanned table of bucket sizes.',
      load: () => import('./S146').then((m) => ({ Component: m.S146, questions: m.questionsS146 })),
    },
    {
      id: '14.7', num: '14.7', title: 'Choice of radix value', pages: '342–344',
      lede: 'Choose a radix width by trading passes against bucket count, coalescing and the size of the table scan.',
      load: () => import('./S147').then((m) => ({ Component: m.S147, questions: m.questionsS147 })),
    },
    {
      id: '14.8', num: '14.8', title: 'Thread coarsening to improve coalescing', pages: '344–345',
      lede: 'Pick how many keys each thread sorts, and say what it buys in bucket size and table size and what it costs in shared memory.',
      load: () => import('./S148').then((m) => ({ Component: m.S148, questions: m.questionsS148 })),
    },
    {
      id: '14.9', num: '14.9', title: 'Other parallel sort methods', pages: '345–346',
      lede: 'Place sorting networks, sample sort and MSD radix sort, pick the library call for your backend, and say why top-k needs a selection, not a sort.',
      load: () => import('./S149').then((m) => ({ Component: m.S149, questions: m.questionsS149 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Three labs on your own GPUs: odd-even and one-bit radix sort, a multi-bit radix sort with local sorting measured against std::stable_sort, and milestone M5, sorted candidates per row.',
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '14.10', title: 'Summary and exercises', pages: '346–348',
      lede: 'What to carry forward, a mixed check, where each exercise is worked, the PDF\'s apparent errata, and milestone M5.',
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
