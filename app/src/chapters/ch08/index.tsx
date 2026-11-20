import type { Chapter } from '../types';

// Condensed tier (PLAN.md): the PDE background is reduced to the start page; the two
// transferable techniques, thread coarsening and register tiling, get a page each.
export const chapter: Chapter = {
  id: 'ch08',
  num: '8',
  title: 'Stencil computation',
  pages: '183–199',
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '183–188',
      lede: 'Say what this chapter adds to your kernel toolkit and why the stencil itself matters little for inference, and check that you have the four ideas from chapters 4 to 7 it builds on.',
      load: () => import('./S80').then((m) => ({ Component: m.S80, questions: m.questionsS80 })),
    },
    {
      id: '8.4', num: '8.4', title: 'A 3D tile and its halo', pages: '187–193',
      lede: "Count what a 3D stencil kernel asks of memory, and see why a cube-shaped tile cannot get far: the thread limit caps its side, the halo takes more than half of it, and each warp's load is split across rows.",
      load: () => import('./S84').then((m) => ({ Component: m.S84, questions: m.questionsS84 })),
    },
    {
      id: '8.5', num: '8.5', title: 'Marching in z, then registers', pages: '193–198',
      lede: 'March a plane of threads through z with three rolling planes and say what each of its two barriers protects, then move the values only one thread reads into registers and say what that buys, what it costs and what it leaves unchanged.',
      load: () => import('./S85').then((m) => ({ Component: m.S85, questions: m.questionsS85 })),
    },
    {
      id: 'labs', num: '', title: 'Lab', pages: null,
      lede: 'One optional lab: write the basic and the register-tiled 3D stencil, test them on every backend, and measure the chapter\'s model on your own GPU.',
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '8.7', title: 'Summary and exercises', pages: '198–199',
      lede: 'What to carry forward, a mixed check on the whole chapter, where each of the book\'s exercises is worked, the PDF\'s apparent slips, and why this chapter has no project milestone.',
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
