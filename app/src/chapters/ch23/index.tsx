import type { Chapter } from '../types';

// Condensed (PLAN.md): the book's section 23.3 (streams, asynchronous copies, pinned memory,
// events) is kept and taught on one GPU; MPI, NCCL and NVSHMEM get a paragraph each on page
// 23.4. One file per section, each loaded on demand and exporting its component and questions.
export const chapter: Chapter = {
  id: 'ch23',
  num: '23',
  title: 'Multi-GPU programming',
  pages: '541–576',
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '541–549',
      lede: "Say what this chapter gives you on one GPU, read the book's running example (a stencil split by rows across GPUs, with halo rows refreshed every step), and check the ideas from chapters 2, 8, 10, 22 and appendix C that it builds on.",
      load: () => import('./S230').then((m) => ({ Component: m.S230, questions: m.questionsS230 })),
    },
    {
      id: '23.3', num: '23.3', title: 'Streams, asynchronous copies and pinned memory', pages: '553–556, 560–563',
      lede: 'Split a job that copies in, computes and copies out across streams so the copies hide behind the kernel, predict how long it takes and what idle time remains, and say why the host buffer must be pinned.',
      load: () => import('./S233').then((m) => ({ Component: m.S233, questions: m.questionsS233 })),
    },
    {
      id: '23.4', num: '23.4', title: 'Events, priorities, and who waits', pages: '556–575',
      lede: 'Order work across streams with events, say what a stream priority can and cannot promise, and trace how MPI, NCCL and NVSHMEM move the waiting from the host into the stream and then into the kernel.',
      load: () => import('./S234').then((m) => ({ Component: m.S234, questions: m.questionsS234 })),
    },
    {
      id: 'labs', num: '', title: 'Lab', pages: null,
      lede: 'One optional lab: split a copy-in, compute, copy-out job across streams and measure what the overlap buys on your GPU against a prediction; on a Mac, keep several buffers in flight so the CPU fills the next chunk while the GPU computes.',
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '23.6', title: 'Summary and exercises', pages: '574–576',
      lede: 'What to carry forward, a mixed check on the whole chapter, where each of the book\'s exercises is worked, the PDF\'s apparent slips and dated claims, and why this chapter has no project milestone.',
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
