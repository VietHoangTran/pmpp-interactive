import type { Chapter } from '../types';

// One file per section, each loaded on demand and exporting its component and its
// questions (the shape of chapter 5, the template).
export const chapter: Chapter = {
  id: 'ch10',
  num: '10',
  title: 'Reduction',
  pages: '221–250',
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '221',
      lede: 'What this chapter is for, where reduction sits under softmax, max-logit and argmax, and a check on what it assumes from chapters 4 to 9.',
      load: () => import('./S100').then((m) => ({ Component: m.S100, questions: m.questionsS100 })),
    },
    {
      id: '10.1', num: '10.1', title: 'Background', pages: '221–223',
      lede: 'Write any reduction as one loop over a binary operator and its identity, and pick a starting value that is safe on real data.',
      load: () => import('./S101').then((m) => ({ Component: m.S101, questions: m.questionsS101 })),
    },
    {
      id: '10.2', num: '10.2', title: 'Reduction trees', pages: '223–227',
      lede: 'Draw the tree a parallel reduction computes, count its work and span, and say what the operator must satisfy for the tree to give the loop\'s answer.',
      load: () => import('./S102').then((m) => ({ Component: m.S102, questions: m.questionsS102 })),
    },
    {
      id: '10.3', num: '10.3', title: 'A simple reduction kernel', pages: '227–230',
      lede: 'Map a reduction tree onto one block of threads, step the in-place kernel, and say what its barrier guarantees.',
      load: () => import('./S103').then((m) => ({ Component: m.S103, questions: m.questionsS103 })),
    },
    {
      id: '10.4', num: '10.4', title: 'Reducing control divergence', pages: '230–234',
      lede: 'Count the divergent warps and the resource efficiency of a reduction, and remove most of both with a three-line change.',
      load: () => import('./S104').then((m) => ({ Component: m.S104, questions: m.questionsS104 })),
    },
    {
      id: '10.5', num: '10.5', title: 'Reducing memory access divergence', pages: '234–235',
      lede: 'Count the memory requests of each kernel from the addresses a warp touches, and see why the convergent kernel coalesces.',
      load: () => import('./S105').then((m) => ({ Component: m.S105, questions: m.questionsS105 })),
    },
    {
      id: '10.6', num: '10.6', title: 'Reducing global memory accesses', pages: '236–237',
      lede: 'Move the tree into shared memory, place its barrier, and count the global traffic that is left.',
      load: () => import('./S106').then((m) => ({ Component: m.S106, questions: m.questionsS106 })),
    },
    {
      id: '10.7', num: '10.7', title: 'Reducing synchronization overhead with warp-level primitives', pages: '237–241',
      lede: 'Exchange values between lanes with shuffles, write a warp reduction with no shared memory and no barrier, and make it compile for CUDA, HIP and Metal.',
      load: () => import('./S107').then((m) => ({ Component: m.S107, questions: m.questionsS107 })),
    },
    {
      id: '10.8', num: '10.8', title: 'Further reducing synchronization overhead with two-stage warp-wide reduction', pages: '241–244',
      lede: 'Reduce every warp in registers, hand off once through shared memory, and argue why more idle lanes can still be faster.',
      load: () => import('./S108').then((m) => ({ Component: m.S108, questions: m.questionsS108 })),
    },
    {
      id: '10.9', num: '10.9', title: 'Reduction for arbitrary length inputs', pages: '244–246',
      lede: 'Reduce any length with many blocks, commit with an atomic, and decide when a second pass is the better design.',
      load: () => import('./S109').then((m) => ({ Component: m.S109, questions: m.questionsS109 })),
    },
    {
      id: '10.10', num: '10.10', title: 'Thread coarsening to reduce overhead', pages: '246–248',
      lede: 'Choose a coarsening factor for a given input and device, and see why the best one leaves just enough blocks to fill the hardware.',
      load: () => import('./S1010').then((m) => ({ Component: m.S1010, questions: m.questionsS1010 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Three labs on your own GPUs: one sum written three ways and measured, a reduction hunt, and milestone M4, softmax and argmax over logits.',
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '10.11', title: 'Summary and exercises', pages: '249–250',
      lede: 'What to carry forward, a mixed check on the whole chapter, where each exercise is worked, the PDF\'s apparent errata, and milestone M4.',
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
