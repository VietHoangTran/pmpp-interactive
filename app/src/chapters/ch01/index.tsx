import type { Chapter } from '../types';

// Planned. The sections are added when the chapter is built (see PLAN.md).
export const chapter: Chapter = {
  id: 'ch01',
  num: '1',
  title: 'Introduction',
  pages: '1–20',
  sections: [
    // Temporary wiring by the builder of section group 1, for its browser checks. The integrator owns this list.
    {
      id: 'start', num: '', title: 'Start here: one budget, two designs', pages: '1–7',
      lede: 'Say why a GPU beats a CPU on many independent operations and loses on one dependent chain, and predict which design finishes a given workload first.',
      load: () => import('./S10').then((m) => ({ Component: m.S10, questions: m.questions10 })),
    },
    // Temporary wiring by the builder of section group 2 (section 1.3), for its browser check. The integrator owns this list.
    {
      id: '1.3', num: '1.3', title: 'Speed-up and what limits it', pages: '7–12',
      lede: 'Put a ceiling on the end-to-end gain from a faster kernel before you write it, and name the four difficulties the rest of the book keeps returning to.',
      load: () => import('./S13').then((m) => ({ Component: m.S13, questions: m.questions13 })),
    },
    // Temporary wiring by the builder of section group 3 (section 1.7), for its browser check. The integrator owns this list.
    {
      id: '1.7', num: '1.7', title: 'The book, this edition, and your path through it', pages: 'xvii–xxvi, 12–18',
      lede: 'Plan the order in which you read the book for your goals, and tell whether a page you find online is talking about this edition.',
      load: () => import('./S17').then((m) => ({ Component: m.S17, questions: m.questions17 })),
    },
  ],
};
