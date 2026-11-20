import type { Chapter } from '../types';

// Condensed: two teaching pages (16.3 covers §16.1 to §16.4 and the synchronisation part of
// §16.8; 16.6 covers §16.5 to §16.8 on Smith-Waterman's table), an optional lab and a
// summary. Each section is loaded on demand and exports its component and its questions.
export const chapter: Chapter = {
  id: 'ch16',
  num: '16',
  title: 'Dynamic programming and wavefront parallelism',
  pages: '373–400',
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '373–374',
      lede: 'Say what a wavefront is and why a recurrence along a sequence needs one, and check the ideas from chapters 2, 4, 5, 6 and 11 that this chapter builds on.',
      load: () => import('./S160').then((m) => ({ Component: m.S160, questions: m.questionsS160 })),
    },
    {
      id: '16.3', num: '16.3', title: 'Wavefronts, and who waits for whom', pages: '374–381, 397–399',
      lede: 'Number the cells of a table by the step at which they can run, say why a Floyd-Warshall step can update its table in place, and choose the cheapest synchronisation (a relaunch, a block barrier, flags, or none) that covers exactly the dependencies.',
      load: () => import('./S163').then((m) => ({ Component: m.S163, questions: m.questionsS163 })),
    },
    {
      id: '16.6', num: '16.6', title: 'Tiles, and the shear that fills them', pages: '382–398',
      lede: 'Map blocks to tiles and threads to cells along anti-diagonal fronts, shear a tile so every thread works at every step, and say what that costs in launches, parallelism and shared-memory banks.',
      load: () => import('./S166').then((m) => ({ Component: m.S166, questions: m.questionsS166 })),
    },
    {
      id: 'labs', num: '', title: 'Lab', pages: null,
      lede: 'One optional lab: fill a dynamic-programming table with square and sheared tiles, test every cell against a CPU reference on every backend, and measure launches, padding and tile width on your GPU.',
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '16.9', title: 'Summary and exercises', pages: '399–400',
      lede: "What to carry forward, a mixed check, where each of the book's exercises is worked, the PDF's apparent slips, what this chapter left out, and why it has no project milestone.",
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
