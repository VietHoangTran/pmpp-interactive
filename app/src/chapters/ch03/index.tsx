import type { Chapter } from '../types';

// Planned. The sections are added when the chapter is built (see PLAN.md).
export const chapter: Chapter = {
  id: 'ch03',
  num: '3',
  title: 'Multidimensional grids and data',
  pages: '45–66',
  sections: [
    // Temporary wiring by the builder of "Before you start" and section 3.1, for the integrator to keep or replace.
    {
      id: 'start', num: '', title: 'Before you start', pages: '45',
      lede: 'What this chapter is for, the three kernels it builds, and a check on what it assumes from chapter 2.',
      load: () => import('./S30').then((m) => ({ Component: m.S30, questions: m.questionsS30 })),
    },
    {
      id: '3.1', num: '3.1', title: 'Multidimensional grid organization', pages: '45–48',
      lede: 'Choose a legal dim3 launch for 1D, 2D or 3D work and say what every thread sees in its four built-ins.',
      load: () => import('./S31').then((m) => ({ Component: m.S31, questions: m.questionsS31 })),
    },
    // Temporary wiring by the section 3.2 builder, for the integrator to keep or replace.
    {
      id: '3.2', num: '3.2', title: 'Mapping threads to multidimensional data', pages: '48–55',
      lede: 'Take any thread of a 2D or 3D grid to its row, column and flat offset, say which threads are surplus, and write the grayscale kernel correctly.',
      load: () => import('./S32').then((m) => ({ Component: m.S32, questions: m.questionsS32 })),
    },
    // Temporary wiring by the section 3.3 builder, for the integrator to keep or replace.
    {
      id: '3.3', num: '3.3', title: 'Image blur: a more complex kernel', pages: '55–59',
      lede: "Write a kernel whose threads loop over a neighbourhood, with one guard for surplus threads and another for the picture's edge, and say what each guard prevents.",
      load: () => import('./S33').then((m) => ({ Component: m.S33, questions: m.questionsS33 })),
    },
    // Temporary wiring by the section 3.4 builder, for the integrator to keep or replace.
    {
      id: '3.4', num: '3.4', title: 'Matrix multiplication', pages: '59–64',
      lede: "Write the one-thread-per-element matmul kernel and its launch for any shape, trace any thread's reads by hand, and say which storage convention your kernel assumes.",
      load: () => import('./S34').then((m) => ({ Component: m.S34, questions: m.questionsS34 })),
    },
  ],
};
