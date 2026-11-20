import type { Chapter } from '../types';

// Planned. The sections are added when the chapter is built (see PLAN.md).
export const chapter: Chapter = {
  id: 'appA',
  num: 'A',
  title: 'Numerical considerations',
  pages: '583–600',
  sections: [
    // TEMP group 4 wiring (SA0, SA6): the integrator replaces this list.
    {
      id: 'start', num: '', title: 'Before you start', pages: '583',
      lede: 'Say where rounding enters one step of LLM inference, what this appendix lets you decide about it, and check what it assumes from chapters 5, 10 and 20.',
      load: () => import('./SA0').then((m) => ({ Component: m.SA0, questions: m.questionsSA0 })),
    },
    {
      id: 'A.6', num: 'A.6', title: 'Algorithm considerations', pages: '593–594',
      lede: 'Predict how summation order and accumulator width change a long sum, and choose sorted, pairwise or compensated summation for a reduction you write.',
      load: () => import('./SA6').then((m) => ({ Component: m.SA6, questions: m.questionsSA6 })),
    },
    // TEMP group 1 wiring (SA1, SA2): the integrator replaces this list.
    {
      id: 'A.1', num: 'A.1', title: 'Floating-point data representation', pages: '583–586',
      lede: 'Turn any bit pattern of FP32, FP16, BF16 or the book\'s toy formats into its value and back, and say what each field buys you.',
      load: () => import('./SA1').then((m) => ({ Component: m.SA1, questions: m.questionsSA1 })),
    },
    {
      id: 'A.2', num: 'A.2', title: 'Representable numbers', pages: '586–590',
      lede: 'Draw every value a small format can hold, see where its gaps are, and predict the rounding error a real format makes near any value.',
      load: () => import('./SA2').then((m) => ({ Component: m.SA2, questions: m.questionsSA2 })),
    },
    // TEMP group 2 wiring (SA3, SA4): the integrator replaces this list.
    {
      id: 'A.3', num: 'A.3', title: 'Precision and special bit patterns in IEEE format', pages: '590–591',
      lede: 'Classify any bit pattern as zero, denormal, normal, infinity or NaN, predict which operations produce infinities and NaNs, and keep a softmax from producing them.',
      load: () => import('./SA3').then((m) => ({ Component: m.SA3, questions: m.questionsSA3 })),
    },
    {
      id: 'A.4', num: 'A.4', title: 'Arithmetic accuracy and rounding', pages: '591–592',
      lede: 'Step an addition through alignment and rounding, measure its error in ULPs under each rounding mode, and say which operations on your GPU are correctly rounded.',
      load: () => import('./SA4').then((m) => ({ Component: m.SA4, questions: m.questionsSA4 })),
    },
    // TEMP group 3 wiring (SA5, SA7): the integrator replaces this list.
    {
      id: 'A.5', num: 'A.5', title: 'Lower-precision formats', pages: '592–593',
      lede: 'Decode FP8 and FP4 values, quantise a block with a shared scale, and work out what fewer bits per weight buy a memory-bound decode step.',
      load: () => import('./SA5').then((m) => ({ Component: m.SA5, questions: m.questionsSA5 })),
    },
    {
      id: 'A.7', num: 'A.7', title: 'Linear solvers and numerical stability', pages: '594–599',
      lede: 'Step Gaussian elimination with one thread per row and its barriers, watch a zero or tiny pivot break it, and say why pivoting costs a max reduction per column.',
      load: () => import('./SA7').then((m) => ({ Component: m.SA7, questions: m.questionsSA7 })),
    },
  ],
};
