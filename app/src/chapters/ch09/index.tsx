import type { Chapter } from '../types';

// Built from the chapter 5 template: one file per section, each loaded on demand and
// exporting its component and its questions.
export const chapter: Chapter = {
  id: 'ch09',
  num: '9',
  title: 'Histogram',
  pages: '201–220',
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '201',
      lede: 'See why this is the first chapter where threads must share their outputs, meet the bug in one line of MLX, and check what the chapter assumes from chapters 4 to 6.',
      load: () => import('./S90').then((m) => ({ Component: m.S90, questions: m.questionsS90 })),
    },
    {
      id: '9.1', num: '9.1', title: 'What a histogram computes', pages: '202–203',
      lede: 'Map any value to its bin, treat an image as a flat array of bytes, and read each bar\'s height as the number of threads that will collide on that bin.',
      load: () => import('./S91').then((m) => ({ Component: m.S91, questions: m.questionsS91 })),
    },
    {
      id: '9.2', num: '9.2', title: 'Atomic operations and a basic histogram kernel', pages: '203–209',
      lede: 'Produce a lost update by choosing how threads interleave, then fix it with cuda::atomic_ref and its HIP and Metal counterparts, choosing the scope and memory order on purpose.',
      load: () => import('./S92').then((m) => ({ Component: m.S92, questions: m.questionsS92 })),
    },
    {
      id: '9.3', num: '9.3', title: 'Latency and throughput of atomic operations', pages: '209–210',
      lede: 'Estimate how many atomic updates per second one location can take from its latency, and predict how much skewed data slows the basic kernel on your GPU.',
      load: () => import('./S93').then((m) => ({ Component: m.S93, questions: m.questionsS93 })),
    },
    {
      id: '9.4', num: '9.4', title: 'Privatisation', pages: '211–214',
      lede: 'Give each block a private histogram, commit it with the right scope, find the bug in a broken one, and decide when the copy fits on chip, including when the bins are a vocabulary.',
      load: () => import('./S94').then((m) => ({ Component: m.S94, questions: m.questionsS94 })),
    },
    {
      id: '9.5', num: '9.5', title: 'Thread coarsening', pages: '214–217',
      lede: 'Give each thread several pixels without breaking coalescing, and count how many private copies and commits coarsening saves.',
      load: () => import('./S95').then((m) => ({ Component: m.S95, questions: m.questionsS95 })),
    },
    {
      id: '9.6', num: '9.6', title: 'Thread-level privatisation', pages: '217–219',
      lede: 'Merge runs of equal values in a register, and predict which data, and which partitioning, make that pay off.',
      load: () => import('./S96').then((m) => ({ Component: m.S96, questions: m.questionsS96 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Two labs and an optional third: privatise into shared memory and measure skew, count tokens over a vocabulary for repetition penalties, and, if you want more practice, hunt the races in two histogram kernels.',
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '9.7', title: 'Summary and exercises', pages: '219–220',
      lede: 'What to carry forward, a mixed check on the whole chapter, where each exercise is worked, the PDF\'s slips, and the project step.',
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
