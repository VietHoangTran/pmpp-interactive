import type { Chapter } from '../types';

// Planned. The sections are added when the chapter is built (see PLAN.md).
export const chapter: Chapter = {
  id: 'ch18',
  num: '18',
  title: 'Graph traversal',
  pages: '425–451',
  sections: [
    // Temporary wiring by the group 1 builder (start, 18.3); the integrator owns this file.
    {
      id: 'start', num: '', title: 'Before you start', pages: '425–431',
      lede: "Say what this chapter adds to an inference engineer's toolkit, read a graph stored as CSR, CSC or COO, label BFS levels by hand, and check what the chapter assumes from chapters 4, 9, 11, 12 and 17.",
      load: () => import('./S180').then((m) => ({ Component: m.S180, questions: m.questionsS180 })),
    },
    {
      id: '18.3', num: '18.3', title: 'Vertices, edges or a frontier', pages: '431–442',
      lede: 'Run one BFS level four ways on a graph you edit, count the work each does, and make a vertex join the next frontier exactly once with compare-and-swap on CUDA, HIP and Metal.',
      load: () => import('./S183').then((m) => ({ Component: m.S183, questions: m.questionsS183 })),
    },
  ],
};
