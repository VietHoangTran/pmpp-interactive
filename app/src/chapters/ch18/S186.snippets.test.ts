import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { CORRECT, VARIANT, kernelBody, levelAtomics, listingLine, siteLine } from '../../sim/bfspriv';
import { EMU_COOP_LIMIT, gridSummary, hostStrip } from '../../sim/gridbfs';
import { region } from '../../sim/histpriv';
import { functionSource } from '../../sim/toolchain';
import { SCENE_GAP } from './OneLaunchScene';

// Section 18.6 loads its listings from labs/ch18/snippets/ at run time, so the code shown is
// always the file. What can drift is what the page says about that code: the snippets of
// code quoted in its prose, the regions and functions it loads, the lines it marks, and the
// numbers it states. This file ties each of those to the files and models it comes from.
// (bfspriv.test.ts and gridbfs.test.ts pin the stepper listings to the same files and run
// them on the emulator.)
const repo = path.resolve(__dirname, '../../../..');
const PRIV = fs.readFileSync(path.join(repo, 'labs/ch18/snippets/s186_privatised_variants.cu'), 'utf8');
const COOP = fs.readFileSync(path.join(repo, 'labs/ch18/snippets/s186_coop_swap.cu'), 'utf8');
const lesson = fs.readFileSync(path.join(__dirname, 'S186.tsx'), 'utf8');
const prose = lesson.replace(/\s+/g, ' ');

describe('code the page loads exists in the snippet files', () => {
  it('the three forms of the reset, with the marked lines the page expects', () => {
    const at = (kernel: string, site: 'reset' | 'clamp', rows: number) => {
      const f = functionSource(PRIV, kernel)!;
      expect(f, kernel).not.toBeNull();
      const lines = f.text.split('\n');
      const k = siteLine(f.text, site);
      return lines.slice(k, k + rows).map((l) => l.trim()).join(' ');
    };
    expect(at('bfsPrivatised', 'reset', 2)).toBe('numCurrFrontier_s_ref.store(PRIVATE_FRONTIER_CAPACITY, cuda::memory_order_relaxed);');
    expect(at('bfsPrivatisedClamp', 'clamp', 2)).toBe('if (numCurrFrontier_s > PRIVATE_FRONTIER_CAPACITY) numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;');
    expect(at('bfsPlainStore', 'reset', 1)).toBe('numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;');
    // The page passes the row counts 2, 2 and 1 for these three tabs.
    expect(lesson).toContain("tab('corrected: atomic store', 'bfsPrivatised', 'reset', 2,");
    expect(lesson).toContain("tab('corrected: clamp', 'bfsPrivatisedClamp', 'clamp', 2,");
    expect(lesson).toContain("tab(\"the book's line 29\", 'bfsPlainStore', 'reset', 1,");
  });
  it('the launch and header regions, and the Exercise 3 kernel', () => {
    const launch = region(COOP, 'launch')!.text;
    expect(launch).toContain('cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, bfsCoop,');
    expect(launch).toContain('int numSMs = deviceProp.multiProcessorCount;');
    expect(launch).toMatch(/void\* kernelArgs\[\] = \{/);
    expect(launch).toMatch(/cudaLaunchCooperativeKernel\(\(void\*\)bfsCoop, numBlocks,\s+numThreadsPerBlock, kernelArgs, 0, 0\);/);
    const header = region(COOP, 'header')!.text;
    expect(header).toContain('#if defined(__HIPCC__)\n#include <hip/hip_cooperative_groups.h>\n#else\n#include <cooperative_groups.h>\n#endif');
    const ex3 = functionSource(COOP, 'bfsCoopEx3')!.text;
    expect(ex3.match(/grid\.sync\(\)/g)).toHaveLength(1);
    expect(ex3).toContain('numPrevFrontier = numFrontier[currLevel];');
  });
});

describe('code quoted in the prose is the code in the files', () => {
  const quoted = (s: string) => expect(prose, s).toContain(s);
  it('the corrected store and the book\'s plain store', () => {
    quoted('Store through the same reference, <code>numCurrFrontier_s_ref</code>, with a relaxed{\' \'} <code>store</code>');
    expect(kernelBody(CORRECT).replace(/\s+/g, ' ')).toContain('numCurrFrontier_s_ref.store(PRIVATE_FRONTIER_CAPACITY, cuda::memory_order_relaxed);');
    quoted('<code>numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;</code>');
    expect(kernelBody(VARIANT['plain-store'].shape)).toContain('numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;');
  });
  it('the commit index the page describes: t, t + blockDim.x, ... from currFrontierStartIdx', () => {
    const body = kernelBody(CORRECT);
    expect(body).toContain('for (unsigned int currFrontierIdx_s = threadIdx.x;');
    expect(body).toContain('currFrontierIdx_s += blockDim.x) {');
    expect(body).toContain('currFrontierStartIdx + currFrontierIdx_s;');
  });
  it('the HIP header and the six-argument form', () => {
    quoted('<code>&lt;hip/hip_cooperative_groups.h&gt;</code>');
    expect(COOP).toContain('#include <hip/hip_cooperative_groups.h>');
    quoted("pass <code>0, 0</code> for shared memory and stream");
    expect(COOP).toContain('kernelArgs, 0, 0);');
  });
  it('the book\'s four-argument call, as the question quotes it (notes/ch18.md, p. 446)', () => {
    const notes = fs.readFileSync(path.join(repo, 'notes/ch18.md'), 'utf8').replace(/\s+/g, ' ');
    expect(notes).toContain('cudaLaunchCooperativeKernel((void*)bfs_kernel, numBlocks, numThreadsPerBlock, kernelArgs);');
    quoted('<code>cudaLaunchCooperativeKernel((void*)bfs_kernel, numBlocks, numThreadsPerBlock, kernelArgs);</code>');
  });
  it('the race panel names the fetch_add and the plain store in the stepper\'s listing', () => {
    const sh = VARIANT['plain-store'].shape;
    const lines = kernelBody(sh, true).split('\n');
    expect(lines[listingLine(sh, 'count') - 1]).toContain('currFrontierIdx_s = numCurrFrontier_s_ref');
    expect(lines[listingLine(sh, 'reset') - 1].trim()).toBe('numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;');
  });
  it('what the page says the emulator does with grid.sync() under <<< >>>, and its own grid limit', () => {
    const emu = fs.readFileSync(path.join(repo, 'labs/common/include/pmpp/emu/emu_device.h'), 'utf8');
    expect(emu).toContain('synchronising the whole grid needs a kernel launched with');
    expect(emu).toContain('cudaLaunchCooperativeKernel, not <<< >>>');
    expect(EMU_COOP_LIMIT).toBe(1024);
    quoted('a cooperative grid of more than 1,024 threads');
  });
});

describe('numbers the page states', () => {
  it('the prediction: 1,998 levels, 1,999 launches, frontiers up to 1,000 of 1,000,000 vertices', () => {
    const g = gridSummary(1000);
    expect([g.levels, g.launches, g.maxFrontier, g.vertices]).toEqual([1998, 1999, 1000, 1_000_000]);
  });
  it('q-atomics: 24; q-hubs: 50,000 / 2 = 25,000 times longer', () => {
    expect(levelAtomics(3000, 24).fig1815).toBe(24);
    expect(50_000 / 2).toBe(25_000);
    quoted('25,000 times longer');
  });
  it('the scene\'s round trip is drawn as the work of 10 vertices, as the Simplification note says', () => {
    expect(SCENE_GAP).toBe(10);
    quoted('A round trip is drawn as the work of 10 vertices');
    expect(hostStrip(gridSummary(16).sizes, SCENE_GAP).parts.filter((p) => p.kind === 'kernel')).toHaveLength(31);
  });
  it('E1 is stated for the range the model checks: up to 5 threads and capacity 4', () => {
    quoted('every interleaving for up to 5 threads and capacity 4');
  });
});

const quoted = (s: string) => expect(prose, s).toContain(s);
