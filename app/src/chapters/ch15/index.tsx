import type { Chapter } from '../types';

// One file per section, each loaded on demand and exporting its component and its
// questions (chapter 5 is the template).
export const chapter: Chapter = {
  id: 'ch15',
  num: '15',
  title: 'Advanced optimizations for matrix multiplication',
  pages: '349–370',
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '349',
      lede: 'See how far your chapter 5 kernel is from the vendor library on your own device, and check what this chapter assumes you remember.',
      load: () => import('./S150').then((m) => ({ Component: m.S150, questions: m.questionsS150 })),
    },
    {
      id: '15.1', num: '15.1', title: 'Background: GEMM and the layouts you will meet', pages: '349–350',
      lede: "Map a linear layer or an attention product onto the book's GEMM, and onto a column-major or row-major library call.",
      load: () => import('./S151').then((m) => ({ Component: m.S151, questions: m.questionsS151 })),
    },
    {
      id: '15.2', num: '15.2', title: 'Data reuse analysis', pages: '350–352',
      lede: 'Compute the arithmetic intensity of any output tile, and say why neither the tile depth nor anything at batch size one can raise it.',
      load: () => import('./S152').then((m) => ({ Component: m.S152, questions: m.questionsS152 })),
    },
    {
      id: '15.3', num: '15.3', title: 'Larger tiles with thread coarsening', pages: '352–357',
      lede: 'Write the 128 x 128 coarsened kernel: say which thread owns which outputs, which elements it loads, and what keeps its tile in registers.',
      load: () => import('./S153').then((m) => ({ Component: m.S153, questions: m.questionsS153 })),
    },
    {
      id: '15.4', num: '15.4', title: 'Register tiling of the input tiles', pages: '357–360',
      lede: 'Interchange the inner loops so each thread reads every shared value once, and count what that saves.',
      load: () => import('./S154').then((m) => ({ Component: m.S154, questions: m.questionsS154 })),
    },
    {
      id: '15.5', num: '15.5', title: 'Coalesced storing of the output tile', pages: '360–361',
      lede: "Re-map threads to four 4 x 4 sub-tiles so that a warp's vector stores coalesce, for 32- and 64-wide warps.",
      load: () => import('./S155').then((m) => ({ Component: m.S155, questions: m.questionsS155 })),
    },
    {
      id: '15.6', num: '15.6', title: 'Eliminating bank conflicts', pages: '361–363',
      lede: 'Find the bank conflicts of a strip read for any leading dimension, and choose the padding that removes them.',
      load: () => import('./S156').then((m) => ({ Component: m.S156, questions: m.questionsS156 })),
    },
    {
      id: '15.7', num: '15.7', title: 'Occupancy considerations', pages: '363–364',
      lede: 'Budget registers and shared memory for a tile shape, and explain why this kernel runs at low occupancy on purpose.',
      load: () => import('./S157').then((m) => ({ Component: m.S157, questions: m.questionsS157 })),
    },
    {
      id: '15.8', num: '15.8', title: 'Software pipelining', pages: '364–368',
      lede: 'Double-buffer and software-pipeline the tile loop with one barrier, and predict when the overlap pays.',
      load: () => import('./S158').then((m) => ({ Component: m.S158, questions: m.questionsS158 })),
    },
    {
      id: '15.9', num: '15.9', title: 'Specialized software and hardware support', pages: '368–369',
      lede: 'Say what tensor cores, asynchronous copies and vendor libraries add, which of them each backend has today, and how far your kernel is from the library.',
      load: () => import('./S159').then((m) => ({ Component: m.S159, questions: m.questionsS159 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Seven labs: build milestone M3 one step at a time, measure every step against lab 5.3 and the vendor library on your devices, then hunt three defects.',
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '15.10', title: 'Summary and exercises', pages: '370',
      lede: "What to carry forward, a mixed check, where the book's exercises and errata are worked, and milestone M3.",
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
