import type { Chapter } from '../types';

// Planned. The sections are added when the chapter is built (see PLAN.md).
export const chapter: Chapter = {
  id: 'ch04',
  num: '4',
  title: 'Compute architecture and scheduling',
  pages: '67–92',
  sections: [
    // Temporary wiring by the builder of section group 1 (start, 4.1, 4.2, 4.3), for its browser checks. The integrator owns this list.
    {
      id: 'start', num: '', title: 'Before you start', pages: '67',
      lede: 'See what the chapter is for, where it sits under a framework call, and check what it assumes you remember from chapters 1 to 3.',
      load: () => import('./S40').then((m) => ({ Component: m.S40, questions: m.questionsS40 })),
    },
    {
      id: '4.1', num: '4.1', title: 'Architecture of a modern GPU', pages: '68',
      lede: 'Name the hardware levels a kernel runs on and say which of their sizes your own device will report.',
      load: () => import('./S41').then((m) => ({ Component: m.S41, questions: m.questionsS41 })),
    },
    {
      id: '4.2', num: '4.2', title: 'Thread block scheduling', pages: '69–70',
      lede: 'Predict how many blocks of a grid are running at any moment and say what being assigned as a whole block gives the threads inside it.',
      load: () => import('./S42').then((m) => ({ Component: m.S42, questions: m.questionsS42 })),
    },
    {
      id: '4.3', num: '4.3', title: 'Synchronisation and transparent scalability', pages: '70–74',
      lede: 'Place a barrier where every thread of a block will reach it, and explain why the same grid runs unchanged on a device of any size.',
      load: () => import('./S43').then((m) => ({ Component: m.S43, questions: m.questionsS43 })),
    },
    // End of group 1's temporary wiring.
    // Temporary wiring by the builder of section group 2 (4.4, 4.5), for its browser checks. The integrator owns this list.
    {
      id: '4.4', num: '4.4', title: 'Warps and SIMD hardware', pages: '74–79',
      lede: 'Work out which threads of a block share a warp for any block shape and any warp width, and say what sharing one means.',
      load: () => import('./S44').then((m) => ({ Component: m.S44, questions: m.questionsS44 })),
    },
    {
      id: '4.5', num: '4.5', title: 'Control divergence', pages: '79–82',
      lede: 'Predict the passes a warp makes at a branch or a data-dependent loop, count the warps a boundary check splits, and judge when that cost matters.',
      load: () => import('./S45').then((m) => ({ Component: m.S45, questions: m.questionsS45 })),
    },
    // End of group 2's temporary wiring.
  ],
};
