import type { Chapter } from '../types';
import { S50, questions50 } from './S50';
import { S51, questions51 } from './S51';
import { S52, questionsS52 } from './S52';
import { S53, questionsS53 } from './S53';
import { S54, questionsS54 } from './S54';
import { S55, questionsS55 } from './S55';
import { S56, questionsS56 } from './S56';
import { Labs, questionsLabs } from './Labs';
import { Wrap, questionsWrap } from './Wrap';

export const ch05: Chapter = {
  id: 'ch05',
  num: '5',
  title: 'Memory architecture and data locality',
  pages: '93–121',
  sections: [
    { id: 'start', num: '', title: 'Before you start', pages: '93', lede: 'What this chapter is for, the kernel it takes apart, and a check on what it assumes you remember.', component: S50 },
    { id: '5.1', num: '5.1', title: 'Memory bandwidth as a performance limiter', pages: '93–97', lede: 'Tell whether a kernel is held back by arithmetic or by memory, and by how much.', component: S51 },
    { id: '5.2', num: '5.2', title: 'Where data can live', pages: '98–104', lede: 'Say which memory each variable of a kernel lands in, who can see it, and how many copies exist.', component: S52 },
    { id: '5.3', num: '5.3', title: 'Tiling for reduced memory traffic', pages: '104–107', lede: 'See the redundant fetches in the naive kernel and how sharing them inside a block removes them.', component: S53 },
    { id: '5.4', num: '5.4', title: 'A tiled matrix multiplication kernel', pages: '108–112', lede: 'Write the tiled kernel, place its two barriers, and say what each one protects.', component: S54 },
    { id: '5.5', num: '5.5', title: 'Boundary checks', pages: '112–115', lede: 'Make the kernel correct for any size and for rectangular matrices.', component: S55 },
    { id: '5.6', num: '5.6', title: 'Memory use and occupancy', pages: '115–118', lede: 'Budget shared memory against the number of threads a device can keep resident.', component: S56 },
    { id: 'labs', num: '', title: 'Labs', pages: null, lede: 'Four labs on your own GPUs: measure, tile, generalise, debug.', component: Labs },
    { id: 'wrap', num: '5.7', title: 'Summary and exercises', pages: '118–121', lede: 'What to carry forward, and a mixed check on the whole chapter.', component: Wrap },
  ],
  questions: {
    ...questions50, ...questions51, ...questionsS52, ...questionsS53, ...questionsS54, ...questionsS55,
    ...questionsS56, ...questionsLabs, ...questionsWrap,
  },
};
