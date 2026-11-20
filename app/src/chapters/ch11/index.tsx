import type { Chapter } from '../types';

// One file per section, each loaded on demand and exporting its component and its
// questions (the shape of chapter 5, the template). The title is the PDF's chapter
// heading, "Scan" (p. 251); the book's subtitle is "And work efficiency in parallel
// algorithms".
export const chapter: Chapter = {
  id: 'ch11',
  num: '11',
  title: 'Scan',
  pages: '251–287',
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '251–252',
      lede: 'What scan does in an inference stack, the chain of kernels this chapter builds, and a check on what it assumes from chapters 4 to 10.',
      load: () => import('./S110').then((m) => ({ Component: m.S110, questions: m.questionsS110 })),
    },
    {
      id: '11.1', num: '11.1', title: 'Background', pages: '252–254',
      lede: 'Compute inclusive and exclusive scans, convert one into the other, and say which one decides whether the token that crosses p is kept.',
      load: () => import('./S111').then((m) => ({ Component: m.S111, questions: m.questionsS111 })),
    },
    {
      id: '11.2', num: '11.2', title: 'Parallel scan with the Kogge-Stone algorithm', pages: '254–258',
      lede: "Trace Kogge-Stone's doubling strides, write its block kernel, and find the race its second barrier prevents.",
      load: () => import('./S112').then((m) => ({ Component: m.S112, questions: m.questionsS112 })),
    },
    {
      id: '11.3', num: '11.3', title: 'Double-buffering to reduce synchronization', pages: '258–261',
      lede: 'Tell the true dependence from the false one, remove the false one with two buffers, and say which buffer ends up holding the result.',
      load: () => import('./S113').then((m) => ({ Component: m.S113, questions: m.questionsS113 })),
    },
    {
      id: '11.4', num: '11.4', title: 'Warp-level primitives to reduce synchronization', pages: '261–265',
      lede: 'Scan a warp with shuffles and no barrier, build a block scan from warp scans with two barriers, and make it compile and run on CUDA, HIP and Metal.',
      load: () => import('./S114').then((m) => ({ Component: m.S114, questions: m.questionsS114 })),
    },
    {
      id: '11.5', num: '11.5', title: 'Work efficiency considerations', pages: '265–267',
      lede: 'Count the additions and steps of sequential, Kogge-Stone and Brent-Kung scans, and predict when the parallel one loses.',
      load: () => import('./S115').then((m) => ({ Component: m.S115, questions: m.questionsS115 })),
    },
    {
      id: '11.6', num: '11.6', title: 'Coarsening to improve work-efficiency', pages: '267–270',
      lede: 'Give each thread a run to scan sequentially, keep loads and stores coalesced, and count what coarsening saves.',
      load: () => import('./S116').then((m) => ({ Component: m.S116, questions: m.questionsS116 })),
    },
    {
      id: '11.7', num: '11.7', title: 'Register tiling to avoid shared memory access latency', pages: '270–272',
      lede: "Keep each thread's run in registers, use shared memory only to coalesce, and remove the bank conflicts its strided accesses cause.",
      load: () => import('./S117').then((m) => ({ Component: m.S117, questions: m.questionsS117 })),
    },
    {
      id: '11.8', num: '11.8', title: 'Memory bandwidth considerations', pages: '272–273',
      lede: "Work out scan's operations per byte, turn your measured bandwidth into a ceiling in elements per second, and judge a kernel by the 80% yardstick.",
      load: () => import('./S118').then((m) => ({ Component: m.S118, questions: m.questionsS118 })),
    },
    {
      id: '11.9', num: '11.9', title: 'Consolidating block segments for a global scan', pages: '273–282',
      lede: "Scan any length by stitching blocks together, count each design's bytes per element, and say on which backends the single-pass kernel is safe.",
      load: () => import('./S119').then((m) => ({ Component: m.S119, questions: m.questionsS119 })),
    },
    {
      id: '11.10', num: '11.10', title: 'Parallel scan with the Brent-Kung algorithm', pages: '282–285',
      lede: "Trace Brent-Kung's up-sweep and down-sweep, count its 2N − 2 − log₂N additions, and argue why Kogge-Stone still wins inside a warp.",
      load: () => import('./S1110').then((m) => ({ Component: m.S1110, questions: m.questionsS1110 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: "Six labs: block scans measured three ways, coarsening against your copy bandwidth, a race hunt, scans of any length (hierarchical everywhere, single-pass on CUDA only), and milestone M5's top-p mask.",
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '11.11', title: 'Summary and exercises', pages: '285–287',
      lede: "What to carry forward, a mixed check, where each exercise is worked, the PDF's apparent errata, and the M5 piece.",
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
