import type { Chapter } from '../types';

// Planned. The sections are added when the chapter is built (see PLAN.md).
export const chapter: Chapter = {
  id: 'ch22',
  num: '22',
  title: 'Algorithm selection and problem formulation',
  pages: '529–540',
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '529',
      lede: 'What this chapter is for, why it has no code of its own, and a check on the earlier chapters it leans on.',
      load: () => import('./S220').then((m) => ({ Component: m.S220, questions: m.questionsS220 })),
    },
    {
      id: '22.1', num: '22.1', title: 'Algorithm selection', pages: '530–531',
      lede: 'Choose between algorithms for one problem by naming what each trades (steps, exposed parallelism, generality, accuracy) on the hardware you actually have.',
      load: () => import('./S221').then((m) => ({ Component: m.S221, questions: m.questionsS221 })),
    },
    {
      id: '22.2', num: '22.2', title: 'Problem decomposition', pages: '532–536',
      lede: 'Decide, for a new computation, whether each thread should own an output or an input, by asking who writes, who conflicts and what needs an atomic.',
      load: () => import('./S222').then((m) => ({ Component: m.S222, questions: m.questionsS222 })),
    },
    {
      id: '22.3', num: '22.3', title: "Application-level considerations: Amdahl's law", pages: '537–538',
      lede: 'Work out the speed-up a whole application gets when only some of its modules run on the GPU, with and without overlap, and what to do about the rest.',
      load: () => import('./S223').then((m) => ({ Component: m.S223, questions: m.questionsS223 })),
    },
    {
      id: '22.4', num: '22.4', title: 'Problem formulation', pages: '538',
      lede: 'Recognise when re-posing the problem beats any amount of kernel tuning, and what the new formulation costs instead.',
      load: () => import('./S224').then((m) => ({ Component: m.S224, questions: m.questionsS224 })),
    },
    {
      id: '22.5', num: '22.5', title: 'Batching: latency versus throughput', pages: '539–540',
      lede: 'Predict where a decode step stops being bandwidth-bound as the batch grows, and say what each user pays in latency for the throughput gained.',
      load: () => import('./S225').then((m) => ({ Component: m.S225, questions: m.questionsS225 })),
    },
  ],
};
