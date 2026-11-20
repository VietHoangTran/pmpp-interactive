import type { Chapter } from '../types';

// Appendix C, built in the shape of chapter 5: one file per section, each loaded on
// demand and exporting its component and its questions.
export const chapter: Chapter = {
  id: 'appC',
  num: 'C',
  title: 'CUDA memories, address spaces, and pointers',
  pages: '623–629',
  sections: [
    {
      id: 'start', num: '', title: 'Before you start', pages: '623',
      lede: "Say why chapter 2's copy-everything model stops being enough, step through the eras of CUDA's memory model on one map, and check what this appendix assumes you remember.",
      load: () => import('./SC0').then((m) => ({ Component: m.SC0, questions: m.questionsSC0 })),
    },
    {
      id: 'C.1', num: 'C.1', title: 'Unified device memory address space', pages: '624',
      lede: 'Write one device helper that takes a plain pointer to shared or global data in CUDA and HIP, and the overloads Metal needs instead.',
      load: () => import('./SC1').then((m) => ({ Component: m.SC1, questions: m.questionsSC1 })),
    },
    {
      id: 'C.2', num: 'C.2', title: 'Zero-copy access to host memory', pages: '624–625',
      lede: 'Decide when a kernel should read host memory in place over the interconnect, by estimating what that does to a memory-bound kernel and when copying first pays off.',
      load: () => import('./SC2').then((m) => ({ Component: m.SC2, questions: m.questionsSC2 })),
    },
    {
      id: 'C.3', num: 'C.3', title: 'Unified virtual address space', pages: '625',
      lede: "Tell from a pointer's value where its memory lives, and say why that alone does not make the pointer safe to dereference or to follow.",
      load: () => import('./SC3').then((m) => ({ Component: m.SC3, questions: m.questionsSC3 })),
    },
    {
      id: 'C.4', num: 'C.4', title: 'Large virtual and physical GPU address spaces', pages: '625–626',
      lede: 'Work out what an address width can reach, see why 32-bit GPU addresses forced special allocations, and connect that to the 32-bit limits that still matter in kernels.',
      load: () => import('./SC4').then((m) => ({ Component: m.SC4, questions: m.questionsSC4 })),
    },
    {
      id: 'C.5', num: 'C.5', title: 'Unified physical address space', pages: '626',
      lede: "Say what a kernel needs before it can load directly from another GPU's memory, what that replaces, and where it stops.",
      load: () => import('./SC5').then((m) => ({ Component: m.SC5, questions: m.questionsSC5 })),
    },
    {
      id: 'C.6', num: 'C.6', title: 'Unified memory', pages: '626–627',
      lede: "Port a CPU function to managed memory, say where its bytes live after every line on each kind of system, and keep CUDA's Unified Memory apart from Apple's unified memory.",
      load: () => import('./SC6').then((m) => ({ Component: m.SC6, questions: m.questionsSC6 })),
    },
    {
      id: 'C.7', num: 'C.7', title: 'Access to the full host virtual address space', pages: '627–628',
      lede: "Check whether a GPU's address translation can reach every host address, and tell 'can name it' apart from 'may access it' by reading three device attributes.",
      load: () => import('./SC7').then((m) => ({ Component: m.SC7, questions: m.questionsSC7 })),
    },
    {
      id: 'C.8', num: 'C.8', title: 'Page fault handling', pages: '628–629',
      lede: 'Predict how many pages a sequence of CPU and GPU accesses moves under bulk migration, fault-driven migration, mapping and replication, and what happens when the data outgrows device memory.',
      load: () => import('./SC8').then((m) => ({ Component: m.SC8, questions: m.questionsSC8 })),
    },
    {
      id: 'C.9', num: 'C.9', title: 'Virtual address space control', pages: '629',
      lede: 'Reserve an address range, back it with physical memory piece by piece, grant access, and use that to grow a buffer such as a KV cache without copying it.',
      load: () => import('./SC9').then((m) => ({ Component: m.SC9, questions: m.questionsSC9 })),
    },
    {
      id: 'labs', num: '', title: 'Labs', pages: null,
      lede: 'Measure on your own devices which of three ways of feeding one kernel wins for dense, repeated and sparse access, after predicting it.',
      load: () => import('./Labs').then((m) => ({ Component: m.Labs, questions: m.questionsLabs })),
    },
    {
      id: 'wrap', num: '', title: 'Summary', pages: '623–629',
      lede: 'Carry forward what each memory model lets you dereference and what it costs, check yourself across the whole appendix, and see where the PDF looks wrong.',
      load: () => import('./Wrap').then((m) => ({ Component: m.Wrap, questions: m.questionsWrap })),
    },
  ],
};
