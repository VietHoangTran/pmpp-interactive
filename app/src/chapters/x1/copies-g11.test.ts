import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

// Labs X1.1 and X1.3 (lab group 11 of module X1) link copies of earlier labs' reference
// solutions, kept in labs/x1/shared-g11/ under distinct file stems so that several can be linked
// into one program. These tests tie each copy to its original: the copy must be a block of
// comment lines followed by the original, changed only as its header says. A change to an
// original then fails here until the copy is regenerated.
const repo = path.resolve(__dirname, '../../../..');
const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');
const SHARED = 'labs/x1/shared-g11/';

/** Every top-level __device__ definition (not a declaration ending in ';') made static. */
const staticHelpers = (src: string) => src.replace(/^__device__ (?!.*;\s*$)/gm, 'static __device__ ');

/** The copy, split into its leading header and the rest, which must equal `expected`. */
function expectCopyOf(copy: string, expected: string) {
  expect(copy.endsWith(expected)).toBe(true);
  const header = copy.slice(0, copy.length - expected.length);
  const lines = header.split('\n').filter((l) => l.length > 0);
  expect(lines.length).toBeGreaterThan(0);
  for (const l of lines) expect(l.startsWith('//')).toBe(true);
  expect(header).toContain('PROVIDED, do not edit');
}

const helperCopies: [string, string][] = [
  ['penalties_g11.cu', 'labs/ch09/lab3_tokencounts/solution/kernel.cu'],
  ['softmax_g11.cu', 'labs/ch10/lab3_softmax_argmax/solution/kernel.cu'],
  ['topp_g11.cu', 'labs/ch11/lab6_topp/solution/kernel.cu'],
  ['topk_g11.cu', 'labs/x1/lab1_topk/solution/kernel.cu'],
];
const verbatimCopies: [string, string][] = [
  ['topk_g11.metal', 'labs/x1/lab1_topk/solution/kernel.metal'],
  ['topk_g11_launch.mm', 'labs/x1/lab1_topk/solution/launch.mm'],
];

describe('labs/x1/shared-g11: copies of earlier reference solutions', () => {
  for (const [copy, orig] of helperCopies) {
    it(`${copy} is ${orig} with its helper functions static, nothing else changed`, () => {
      const src = read(orig);
      const c = read(SHARED + copy);
      expectCopyOf(c, staticHelpers(src));
      expect(c).toContain(orig.replace('labs/', ''));
      // Nothing left with external linkage but kernels and launchers.
      expect(staticHelpers(src).match(/^__device__ .*\{/gm)).toBeNull();
    });
  }
  it('sortrows_g11.cu is lab 14.3\'s solution with laneIdx and warpIdx static, nothing else changed', () => {
    const src = read('labs/ch14/lab3_sortrows/solution/kernel.cu');
    const lane = '__device__ __forceinline__ unsigned int laneIdx()';
    const warp = '__device__ __forceinline__ unsigned int warpIdx()';
    expect(src.split(lane).length).toBe(2);
    expect(src.split(warp).length).toBe(2);
    expectCopyOf(read(SHARED + 'sortrows_g11.cu'), src.replace(lane, 'static ' + lane).replace(warp, 'static ' + warp));
  });
  for (const [copy, orig] of verbatimCopies) {
    it(`${copy} is ${orig}, unchanged`, () => {
      expectCopyOf(read(SHARED + copy), read(orig));
    });
  }
  it('the labs link the copies and the earlier labs\' Metal hosts they say they do', () => {
    const l1 = read('labs/x1/lab1_topk/CMakeLists.txt');
    expect(l1).toContain('shared-g11/sortrows_g11.cu');
    expect(l1).toContain('ch14/lab3_sortrows/solution/launch.mm');
    const l3 = read('labs/x1/lab3_sampler/CMakeLists.txt');
    for (const f of ['penalties_g11.cu', 'softmax_g11.cu', 'topp_g11.cu', 'sortrows_g11.cu', 'topk_g11.cu', 'minp_g11.cu',
      'topk_g11_launch.mm', 'minp_g11_launch.mm'])
      expect(l3).toContain('shared-g11/' + f);
    for (const lab of ['ch09/lab3_tokencounts', 'ch10/lab3_softmax_argmax', 'ch11/lab6_topp', 'ch14/lab3_sortrows'])
      expect(l3).toContain(`${lab}/solution/launch.mm`);
  });
});
