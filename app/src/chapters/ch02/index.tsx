import type { Chapter } from '../types';

// One file per section, each loaded on demand and exporting its component and its
// questions, in the shape of chapter 5 (the template).
export const chapter: Chapter = {
  id: 'ch02',
  num: '2',
  title: 'Heterogeneous data-parallel computing',
  pages: '21–44',
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '21',
      lede: 'Know what this chapter builds, see the one-line kernel you will be able to write by the end, and check what it assumes from chapter 1.',
      load: () => import('./S20').then((m) => ({ Component: m.S20, questions: m.questionsS20 })),
      vietnamese: () => import('./S20?pmpp-language=vi').then((m) => ({ Component: m.S20, questions: m.questionsS20 })),
    },
    {
      id: '2.1', num: '2.1', title: 'Data parallelism', pages: '21–23',
      lede: 'Look at an operation and say whether its outputs can be computed independently, which is the property every kernel in this chapter relies on.',
      load: () => import('./S21').then((m) => ({ Component: m.S21, questions: m.questionsS21 })),
      vietnamese: () => import('./S21?pmpp-language=vi').then((m) => ({ Component: m.S21, questions: m.questionsS21 })),
    },
    {
      id: '2.2', num: '2.2', title: 'Host and device: the shape of a CUDA program', pages: '24–25',
      lede: 'Read a CUDA program as one host thread that launches grids of device threads, and say what has and has not happened when a launch statement returns.',
      load: () => import('./S22').then((m) => ({ Component: m.S22, questions: m.questionsS22 })),
      vietnamese: () => import('./S22?pmpp-language=vi').then((m) => ({ Component: m.S22, questions: m.questionsS22 })),
    },
    {
      id: '2.3', num: '2.3', title: 'Vector addition and the host stub', pages: '26–28',
      lede: 'Lay out the three parts of a host function that moves a loop to the device, and count what each round trip across the host/device boundary costs.',
      load: () => import('./S23').then((m) => ({ Component: m.S23, questions: m.questionsS23 })),
      vietnamese: () => import('./S23?pmpp-language=vi').then((m) => ({ Component: m.S23, questions: m.questionsS23 })),
    },
    {
      id: '2.4', num: '2.4', title: 'Device memory and data transfer', pages: '28–32',
      lede: 'Write and check the allocation, copy and free calls of a host stub, and say exactly which of them disappear on Apple silicon and what you still owe in their place.',
      load: () => import('./S24').then((m) => ({ Component: m.S24, questions: m.questionsS24 })),
    },
    {
      id: '2.5', num: '2.5', title: 'Kernel functions and threading', pages: '32–37',
      lede: "Write a one-dimensional kernel: compute each thread's element from its block and thread indices, guard the surplus threads, and say which variables every thread owns privately.",
      load: () => import('./S25').then((m) => ({ Component: m.S25, questions: m.questionsS25 })),
    },
    {
      id: '2.6', num: '2.6', title: 'Calling a kernel', pages: '37–39',
      lede: 'Launch a kernel for any n: compute the block count without the integer-division bug, check the launch, and explain why the result cannot depend on the order blocks run in.',
      load: () => import('./S26').then((m) => ({ Component: m.S26, questions: m.questionsS26 })),
    },
    {
      id: '2.7', num: '2.7', title: 'Compilation', pages: '39–40',
      lede: 'Follow one source file through the compiler on each backend, and say for any function who compiles it, who may call it and who runs it.',
      load: () => import('./S27').then((m) => ({ Component: m.S27, questions: m.questionsS27 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Four labs on your own GPUs: write vector addition end to end, find out where its time goes, build the elementwise operations of the project, and repair a stub with planted defects.',
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '2.8', title: 'Summary and exercises', pages: '40–43',
      lede: 'What to carry forward, a mixed check on the whole chapter, where each book exercise is worked, and where the page and today\'s documentation disagree.',
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
