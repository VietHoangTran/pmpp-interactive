import type { Chapter } from '../types';

// Module X1 is not in the book: every section is ours, so no section carries book pages.
// Built in the shape of chapter 5: one file per section, each loaded on demand and
// exporting its component and its questions.
export const chapter: Chapter = {
  id: 'x1',
  num: 'X1',
  title: 'Sampling kernels',
  pages: 'not in the book',
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: null,
      lede: 'See where the sampler sits after the model’s last matmul, which earlier lab supplies each of its stages, and check what this module assumes from chapters 9 to 14.',
      load: () => import('./SX10').then((m) => ({ Component: m.SX10, questions: m.questionsSX10 })),
    },
    {
      id: 'X1.1', num: 'X1.1', title: 'The sampler chain', pages: null,
      lede: 'Name every stage between logits and a token, say which stages depend on their order, and state the order and tie rule that milestone M5 implements.',
      load: () => import('./SX11').then((m) => ({ Component: m.SX11, questions: m.questionsSX11 })),
    },
    {
      id: 'X1.2', num: 'X1.2', title: 'Temperature, softmax and greedy', pages: null,
      lede: 'Fold temperature into lab 10.3’s softmax, predict how it reshapes a distribution, and say what greedy decoding returns on a tie on each backend.',
      load: () => import('./SX12').then((m) => ({ Component: m.SX12, questions: m.questionsSX12 })),
    },
    {
      id: 'X1.3', num: 'X1.3', title: 'Penalties from token counts', pages: null,
      lede: 'Apply repetition, frequency and presence penalties from lab 9.3’s counts in coalesced, memory-bound elementwise passes, and say which sign rule each project uses.',
      load: () => import('./SX13').then((m) => ({ Component: m.SX13, questions: m.questionsSX13 })),
    },
    {
      id: 'X1.4', num: 'X1.4', title: 'Top-k is a selection, not a sort', pages: null,
      lede: 'Find the k largest of 150,000 logits without sorting them, by an 8-bit radix select, and predict how much reading and writing that saves over a full sort.',
      load: () => import('./SX14').then((m) => ({ Component: m.SX14, questions: m.questionsSX14 })),
    },
    {
      id: 'X1.5', num: 'X1.5', title: 'Top-p over a sorted row', pages: null,
      lede: 'Compose lab 14.3’s sort, lab 10.3’s softmax and lab 11.6’s scan into top-p, state its boundary and tie rule exactly, and decide how much of the row you really need to sort.',
      load: () => import('./SX15').then((m) => ({ Component: m.SX15, questions: m.questionsSX15 })),
    },
    {
      id: 'X1.6', num: 'X1.6', title: 'Min-p as a filter', pages: null,
      lede: 'Turn min-p into one comparison against max + ln p, run it as lab 12.4’s stable filter or as a prefix count on a sorted row, and predict how temperature placement changes what it keeps.',
      load: () => import('./SX16').then((m) => ({ Component: m.SX16, questions: m.questionsSX16 })),
    },
    {
      id: 'X1.7', num: 'X1.7', title: 'Random numbers every backend agrees on', pages: null,
      lede: 'Derive one uniform number per row and step from (seed, stream, step) with Philox4x32-10, check it against published test vectors, and convert its bits to a float that is identical on CUDA, HIP, Metal and the CPU.',
      load: () => import('./SX17').then((m) => ({ Component: m.SX17, questions: m.questionsSX17 })),
    },
    {
      id: 'X1.8', num: 'X1.8', title: 'The draw: inverse CDF over a scan', pages: null,
      lede: 'Pick the token where u lands in the running sum of the survivors’ weights with one block per row, make it safe at both ends of [0, 1), and test it for exact agreement across backends despite floating-point differences.',
      load: () => import('./SX18').then((m) => ({ Component: m.SX18, questions: m.questionsSX18 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Four labs on your own GPUs: top-k by selection against a full sort, top-p and min-p composed and then fused, the M5 sampler end to end with a fixed seed, and a hunt for three defects in a fused filter.',
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '', title: 'Summary', pages: null,
      lede: 'What to carry forward, a mixed check, where every exercise of the module is worked, where the sources disagree, and milestone M5.',
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
