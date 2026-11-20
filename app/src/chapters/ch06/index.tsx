import type { Chapter } from '../types';

// One file per section, each loaded on demand and exporting its component and its
// questions (chapter 5 is the template).
export const chapter: Chapter = {
  id: 'ch06',
  num: '6',
  title: 'Performance considerations',
  pages: '123–155',
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '123–124',
      lede: 'See what this chapter adds to your tiled GEMM, where it sits under a linear layer, and check what it assumes from chapters 3 to 5.',
      load: () => import('./S60').then((m) => ({ Component: m.S60, questions: m.questionsS60 })),
    },
    {
      id: '6.1', num: '6.1', title: 'Global memory access coalescing', pages: '124–132',
      lede: "Look at an index expression and say how many memory transactions one warp's load costs, then fix a bad one by remapping threads or by turning the corner through shared memory.",
      load: () => import('./S61').then((m) => ({ Component: m.S61, questions: m.questionsS61 })),
    },
    {
      id: '6.2', num: '6.2', title: 'Hiding memory latency', pages: '132–137',
      lede: 'Explain why DRAM needs many requests in flight across channels and banks, and why occupancy is a bandwidth tool as well as a latency tool.',
      load: () => import('./S62').then((m) => ({ Component: m.S62, questions: m.questionsS62 })),
    },
    {
      id: '6.3', num: '6.3', title: 'Vector loads and stores', pages: '138–139',
      lede: 'Move 16 bytes per instruction with correct boundaries and alignment, and judge when it pays.',
      load: () => import('./S63').then((m) => ({ Component: m.S63, questions: m.questionsS63 })),
    },
    {
      id: '6.4', num: '6.4', title: 'Shared memory bank conflicts', pages: '139–141',
      lede: 'Compute which bank each lane hits for any stride or tile shape, and remove a conflict with one column of padding.',
      load: () => import('./S64').then((m) => ({ Component: m.S64, questions: m.questionsS64 })),
    },
    {
      id: '6.5', num: '6.5', title: 'Thread coarsening', pages: '141–143',
      lede: 'Decide whether giving each thread more work removes a real overhead, and pick a factor that does not starve the device.',
      load: () => import('./S65').then((m) => ({ Component: m.S65, questions: m.questionsS65 })),
    },
    {
      id: '6.6', num: '6.6', title: 'Loop unrolling', pages: '144–146',
      lede: 'Predict what unrolling does to branches, stalls and where a local array lives.',
      load: () => import('./S66').then((m) => ({ Component: m.S66, questions: m.questionsS66 })),
    },
    {
      id: '6.7', num: '6.7', title: 'Double buffering', pages: '146–147',
      lede: 'Remove the barrier that guards a false dependence by alternating two buffers, and show the result is race-free.',
      load: () => import('./S67').then((m) => ({ Component: m.S67, questions: m.questionsS67 })),
    },
    {
      id: '6.8', num: '6.8', title: 'A checklist of optimisations', pages: '147–153',
      lede: 'Go from a bottleneck to the optimisations that relieve it, with what each one costs and where the book uses it again.',
      load: () => import('./S68').then((m) => ({ Component: m.S68, questions: m.questionsS68 })),
    },
    {
      id: '6.9', num: '6.9', title: 'Optimisation strategy', pages: '153–154',
      lede: 'Find the bottleneck, apply one fix, measure, and use the roofline to decide when to stop.',
      load: () => import('./S69').then((m) => ({ Component: m.S69, questions: m.questionsS69 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Five labs on your own GPUs: see coalescing and bank conflicts in your timings, vectorise a kernel, tune the linear layer (milestone M2), and double-buffer with a race detector watching.',
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '6.10', title: 'Summary and exercises', pages: '154–155',
      lede: 'What to carry forward, a mixed check on situations the sections did not show, and where each of the book\'s exercises is worked.',
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
