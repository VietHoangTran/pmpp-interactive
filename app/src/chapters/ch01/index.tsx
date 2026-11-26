import type { Chapter } from '../types';

// A condensed chapter: the book's chapter 1 has no code and no exercises, so it has
// no labs page. One file per section, each loaded on demand and exporting its
// component and its questions (chapter 5 is the template).
export const chapter: Chapter = {
  id: 'ch01',
  num: '1',
  title: 'Introduction',
  pages: '1–20',
  sections: [
    {
      id: 'start', num: '', title: 'Start here: one budget, two designs', pages: '1–7',
      lede: 'Say why a GPU beats a CPU on many independent operations and loses on one dependent chain, and predict which design finishes a given workload first.',
      vietnamese: true,
      load: () => import('./S10').then((m) => ({ Component: m.S10, questions: m.questions10 })),
    },
    {
      id: '1.3', num: '1.3', title: 'Speed-up and what limits it', pages: '7–12',
      lede: 'Put a ceiling on the end-to-end gain from a faster kernel before you write it, and name the four difficulties the rest of the book keeps returning to.',
      load: () => import('./S13').then((m) => ({ Component: m.S13, questions: m.questions13 })),
      vietnamese: () => import('./S13?pmpp-language=vi').then((m) => ({ Component: m.S13, questions: m.questions13 })),
    },
    {
      id: '1.7', num: '1.7', title: 'The book, this edition, and your path through it', pages: 'xvii–xxvi, 12–18',
      lede: 'Plan the order in which you read the book for your goals, and tell whether a page you find online is talking about this edition.',
      load: () => import('./S17').then((m) => ({ Component: m.S17, questions: m.questions17 })),
      vietnamese: () => import('./S17?pmpp-language=vi').then((m) => ({ Component: m.S17, questions: m.questions17 })),
    },
    {
      id: 'wrap', num: '', title: 'Summary and check', pages: '1–18',
      lede: "What to carry into chapter 2, a mixed check on the whole chapter, and the places where the book's first chapter is dated or the PDF slips.",
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
      vietnamese: () => import('./Wrap?pmpp-language=vi').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
