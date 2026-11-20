import type { Chapter } from '../types';

// Chapter 17 is condensed: two teaching pages built around the book's format comparison,
// one optional lab, and a summary. Each section loads on demand and exports its component
// and its questions.
export const chapter: Chapter = {
  id: 'ch17',
  num: '17',
  title: 'Sparse matrix computation',
  pages: '401–423',
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '401–404',
      lede: 'Say what SpMV computes and why this course keeps only its techniques, judge when a sparse layout reads fewer bytes than a dense one, and check the ideas from chapters 4, 6, 9 and 11 and appendix A that the chapter builds on.',
      load: () => import('./S170').then((m) => ({ Component: m.S170, questions: m.questionsS170 })),
    },
    {
      id: '17.2', num: '17.2', title: 'Formats decide the kernel', pages: '402–414, 418–421',
      lede: 'Turn a small sparse matrix into COO, CSR, ELL and CSC, give each the SpMV thread mapping it implies, and say for each whether it needs atomics, whether its loads coalesce and where its warps diverge.',
      load: () => import('./S172').then((m) => ({ Component: m.S172, questions: m.questionsS172 })),
    },
    {
      id: '17.5', num: '17.5', title: 'Buying regularity: padding, spilling, sorting', pages: '412–422',
      lede: "Price ELL's padding against CSR's divergence on rows of uneven length, choose a hybrid split or a sort, say when the preprocessing pays for itself, and recognise the same moves in mixture-of-experts routing.",
      load: () => import('./S175').then((m) => ({ Component: m.S175, questions: m.questionsS175 })),
    },
    {
      id: 'labs', num: '', title: 'Lab', pages: null,
      lede: 'One optional lab: write CSR and ELL SpMV, test them against a CPU reference on every backend, then measure them on matrices whose row lengths are uniform, spread, sorted and skewed, with a hybrid split, after predicting the outcome.',
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '17.8', title: 'Summary and exercises', pages: '421–422',
      lede: "What to carry forward, a mixed check, where each of the book's five exercises is worked, the PDF's apparent slips in this chapter, and why the chapter has no project milestone.",
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
