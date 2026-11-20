import type { Chapter } from '../types';

// Condensed tier (PLAN.md): the molecular-modelling background is reduced to the start page;
// the transferable decisions, scatter against gather with coarsening-aware ownership (§§21.2
// to 21.4) and cutoff binning (§21.5), get a page each.
export const chapter: Chapter = {
  id: 'ch21',
  num: '21',
  title: 'Electrostatic potential map',
  pages: '513–528',
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '513–514',
      lede: 'Say what this chapter adds to your toolkit (four decisions about one all-pairs sum, not molecular modelling) and check the coalescing, constant-memory, atomics, coarsening and divergence ideas it builds on.',
      load: () => import('./S210').then((m) => ({ Component: m.S210, questions: m.questionsS210 })),
    },
    {
      id: '21.2', num: '21.2', title: 'Who writes where: scatter, gather and coarsening', pages: '515–522',
      lede: 'Choose the gather form of an all-pairs sum so no thread needs an atomic, give each thread several outputs so one load serves all of them, and hand those outputs out so that every write statement stays coalesced.',
      load: () => import('./S212').then((m) => ({ Component: m.S212, questions: m.questionsS212 })),
    },
    {
      id: '21.5', num: '21.5', title: 'Cutoff binning: making all-pairs scale', pages: '523–527',
      lede: 'Make an all-pairs computation scale by cutting it off at a radius: bin the inputs, give each block a list of neighbour bins computed once, pad bins to one size, and send the overflow down a slower path.',
      load: () => import('./S215').then((m) => ({ Component: m.S215, questions: m.questionsS215 })),
    },
    {
      id: 'labs', num: '', title: 'Lab', pages: null,
      lede: 'One optional lab: write the scatter, gather and coarsened kernels for one slice of the potential map, test them on every backend, and make two predictions before you measure them on your own GPU.',
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '21.6', title: 'Summary and exercises', pages: '527–528',
      lede: 'What to carry forward, a mixed check on the whole chapter, where each of the book\'s five exercises is worked, the PDF\'s apparent slips, and why the chapter has no project milestone.',
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
