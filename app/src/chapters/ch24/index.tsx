import type { Chapter } from '../types';

// A condensed chapter: the book's chapter 24 has no code and no exercises, so it has no
// labs page. The whole lesson is on the start page; the wrap page carries it forward.
// One file per section, each loaded on demand (chapter 5 is the template).
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
    {
      id: 'wrap', num: '', title: 'Summary and where next', pages: '577–580',
      lede: 'Carry three habits out of the book\'s last chapter, check them on new numbers, and pick the module that serves your next goal.',
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
