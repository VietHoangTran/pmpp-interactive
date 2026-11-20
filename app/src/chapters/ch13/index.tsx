import type { Chapter } from '../types';

// Condensed: one teaching page (13.3 covers §13.3 to §13.8 of the book), an optional
// lab and a summary. Each section is loaded on demand and exports its component and
// its questions.
export const chapter: Chapter = {
  id: 'ch13',
  num: '13',
  title: 'Merge',
  pages: '303–328',
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '303–306',
      lede: 'Say why merge is the first pattern where a thread cannot work out its inputs from its index, and check the ideas from chapters 2, 5 and 6 that this chapter depends on.',
      load: () => import('./S130').then((m) => ({ Component: m.S130, questions: m.questionsS130 })),
    },
    {
      id: '13.3', num: '13.3', title: 'Co-rank: split the output, search for the inputs', pages: '306–327',
      lede: 'Use a binary search to find where any slice of the merged output begins in each input, use that to give every thread an equal share of a merge, and say what staging the inputs in shared memory buys and what it wastes.',
      load: () => import('./S133').then((m) => ({ Component: m.S133, questions: m.questionsS133 })),
    },
    {
      id: 'labs', num: '', title: 'Lab', pages: null,
      lede: 'One optional lab: merge two sorted arrays of key–value pairs with co-rank partitioning, first straight from global memory and then through shared memory, tested for stability against std::merge on every backend.',
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '13.9', title: 'Summary and exercises', pages: '327–328',
      lede: "What to carry forward, a mixed check, where each of the book's four exercises is worked, the PDF's apparent slips (most of them in the circular-buffer listings this course leaves out), and why the chapter has no project milestone.",
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
