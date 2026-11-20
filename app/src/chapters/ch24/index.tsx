import type { Chapter } from '../types';

// Planned. The sections are added when the chapter is built (see PLAN.md).
export const chapter: Chapter = {
  id: 'ch24',
  num: '24',
  title: 'Conclusion and future outlook',
  pages: '577–580',
  sections: [
    {
      id: 'start', num: '', title: "The book's goals and its last table", pages: '577–580',
      lede: "Place every chapter in the book's three steps, check the closing chapter's one table against its own arithmetic, and read that table the way a kernel writer would.",
      load: () => import('./S240').then((m) => ({ Component: m.S240, questions: m.questions240 })),
    },
  ],
};
