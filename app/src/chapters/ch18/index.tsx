import type { Chapter } from '../types';

// Chapter 18, condensed: a start page, two section pages (18.3 covers §18.3 to §18.5, 18.6
// covers §18.6 to §18.8), one optional lab and the summary. Each section loads on demand.
export const chapter: Chapter = {
  id: 'ch18',
  num: '18',
  title: 'Graph traversal',
  pages: '425–451',
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '425–431',
      lede: "Say what this chapter adds to an inference engineer's toolkit, read a graph stored as CSR, CSC or COO, label BFS levels by hand, and check what the chapter assumes from chapters 4, 9, 11, 12 and 17.",
      load: () => import('./S180').then((m) => ({ Component: m.S180, questions: m.questionsS180 })),
    },
    {
      id: '18.3', num: '18.3', title: 'Vertices, edges or a frontier', pages: '431–442',
      lede: 'Run one BFS level four ways on four small graphs, count the work each does, and make a vertex join the next frontier exactly once with compare-and-swap on CUDA, HIP and Metal.',
      load: () => import('./S183').then((m) => ({ Component: m.S183, questions: m.questionsS183 })),
    },
    {
      id: '18.6', num: '18.6', title: 'Private queues and one launch for every level', pages: '442–449',
      lede: "Build each block's share of the next frontier in shared memory with an overflow path and one global commit, fix the book's one racy line, and run every level in a single cooperative launch on CUDA and HIP, knowing what Metal does instead.",
      load: () => import('./S186').then((m) => ({ Component: m.S186, questions: m.questionsS186 })),
    },
    {
      id: 'labs', num: '', title: 'Lab', pages: null,
      lede: 'One optional lab: a frontier BFS with a privatised queue, tested against a CPU BFS on every backend and under the race detector, then timed against the fixed cost of one launch per level; on CUDA and HIP, the single-launch version too.',
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '18.9', title: 'Summary and exercises', pages: '449–451',
      lede: 'What to carry forward, a mixed check, where the book\'s three exercises are worked, the PDF\'s apparent slips, and why the chapter has no project milestone.',
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
