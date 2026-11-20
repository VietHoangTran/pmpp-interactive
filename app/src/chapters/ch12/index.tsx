import type { Chapter } from '../types';

// One file per section, each loaded on demand and exporting its component and its
// questions (the shape of chapter 5, the template).
export const chapter: Chapter = {
  id: 'ch12',
  num: '12',
  title: 'Filter',
  pages: '289–302',
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '289',
      lede: 'See where compaction sits in an LLM sampler, why frameworks avoid it, and check what the chapter assumes from chapters 4, 6, 9, 10 and 11.',
      load: () => import('./S120').then((m) => ({ Component: m.S120, questions: m.questionsS120 })),
    },
    {
      id: '12.1', num: '12.1', title: 'Background', pages: '289–290',
      lede: 'Say what a filter must produce, and decide for a given job whether it has to keep input order and whether it can work in place.',
      load: () => import('./S121').then((m) => ({ Component: m.S121, questions: m.questionsS121 })),
    },
    {
      id: '12.2', num: '12.2', title: 'A simple parallel unstable filter', pages: '290–291',
      lede: 'Write the one-atomic-per-key filter, explain why its output order changes between runs, and estimate how many atomics its single counter has to absorb.',
      load: () => import('./S122').then((m) => ({ Component: m.S122, questions: m.questionsS122 })),
    },
    {
      id: '12.3', num: '12.3', title: 'Coalescing atomic operations with warp-level primitives', pages: '291–295',
      lede: 'Give every passing lane its output slot with one vote, one population count and one atomic per warp, and write it so the same idea compiles for CUDA, HIP and Metal.',
      load: () => import('./S123').then((m) => ({ Component: m.S123, questions: m.questionsS123 })),
    },
    {
      id: '12.4', num: '12.4', title: 'Privatization', pages: '295–297',
      lede: 'Give each block a private list and counter in shared memory, place the three barriers, and commit the block\'s survivors with one atomic and coalesced stores.',
      load: () => import('./S124').then((m) => ({ Component: m.S124, questions: m.questionsS124 })),
    },
    {
      id: '12.5', num: '12.5', title: 'A simple parallel stable filter', pages: '297–298',
      lede: 'Turn keep flags into output slots with an exclusive scan, compute a block\'s slots from warp votes, and say why a stable filter cannot take its slot from a counter.',
      load: () => import('./S125').then((m) => ({ Component: m.S125, questions: m.questionsS125 })),
    },
    {
      id: '12.6', num: '12.6', title: 'Improving memory coalescing with shared memory and thread coarsening', pages: '298–300',
      lede: 'Stage each block\'s survivors in shared memory so consecutive threads write consecutive slots, and pick a coarsening factor from what it does to chunk length, store instructions and the scan.',
      load: () => import('./S126').then((m) => ({ Component: m.S126, questions: m.questionsS126 })),
    },
    {
      id: '12.7', num: '12.7', title: 'In-place stable filter', pages: '300',
      lede: 'State the ordering an in-place filter needs, show why the single-pass scan provides it on CUDA, and choose a design that is documented-safe on HIP and Metal.',
      load: () => import('./S127').then((m) => ({ Component: m.S127, questions: m.questionsS127 })),
    },
    {
      id: '12.8', num: '12.8', title: 'Related patterns', pages: '301',
      lede: 'Recognise duplicate removal and row or column removal and insertion as one-directional data movement, compute positions without a scan when the index decides, and say which way the in-place ordering runs.',
      load: () => import('./S128').then((m) => ({ Component: m.S128, questions: m.questionsS128 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Five labs on your own GPUs: the unstable filter three ways across a sweep of fractions, a filter hunt, a stable filter by scan, milestone M5\'s candidate compaction, and an optional in-place single pass.',
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '12.9', title: 'Summary and exercises', pages: '302',
      lede: 'What to carry forward, a mixed check on the whole chapter, where each exercise is worked, the PDF\'s apparent errata, and the M5 piece.',
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
