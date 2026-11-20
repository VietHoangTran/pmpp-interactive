import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

// Section 3.1 and the "Before you start" page load their own code from
// labs/ch03/snippets at run time, so there is no copy to drift. What can drift is
// (1) the line numbers the lessons point at and (2) the one third-party quote, which
// is copied into S31.tsx from notes/today/ch03-facts.md (item D1).
const repo = path.resolve(__dirname, '../../../..');
const lines = (name: string) => fs.readFileSync(path.join(repo, 'labs/ch03/snippets', name), 'utf8').split('\n');
const lesson = fs.readFileSync(path.join(__dirname, 'S31.tsx'), 'utf8');
const constant = (name: string) => {
  const m = new RegExp('const ' + name + ' = `([\\s\\S]*?)`;').exec(lesson);
  if (!m) throw new Error(`no ${name} in S31.tsx`);
  return m[1];
};

describe('lines the lessons refer to by number', () => {
  it('vec_add_1d.cu: line 4 is the mapping, line 5 the guard', () => {
    const l = lines('vec_add_1d.cu');
    expect(l[3]).toContain('int i = blockIdx.x * blockDim.x + threadIdx.x;');
    expect(l[4]).toContain('if (i < n)');
    expect(l.join('\n')).toContain('int blocks = (n + threads - 1) / threads;');
  });
  it('launch_1d.cu: lines 5 and 6 are the two dim3 values, line 12 the shorthand', () => {
    const l = lines('launch_1d.cu');
    expect(l[4]).toContain('dim3 dimGrid(32, 1, 1);');
    expect(l[5]).toContain('dim3 dimBlock(128, 1, 1);');
    expect(l[11]).toContain('vecAddKernel<<<32, 128>>>');
  });
  it('launch_fig31.cu: line 9 numbers the thread x fastest, lines 15 and 16 are the launch of Fig. 3.1', () => {
    const l = lines('launch_fig31.cu');
    expect(l[8]).toContain('int thread = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;');
    expect(l[14]).toContain('dim3 dimGrid(2, 2, 1);');
    expect(l[15]).toContain('dim3 dimBlock(4, 2, 2);');
  });
  it('launch_checked.cu: line 11 launches, line 12 asks for the error', () => {
    const l = lines('launch_checked.cu');
    expect(l[10]).toContain('fillOnes<<<1, dimBlock>>>(out_d, n);');
    expect(l[11]).toContain('return cudaGetLastError();');
    expect(l.join('\n')).toContain('dim3(32, 32, 2)');
    expect(l.join('\n')).toContain('cudaErrorInvalidConfiguration');
  });
});

describe('the llama.cpp quote', () => {
  const facts = fs.readFileSync(path.join(repo, 'notes/today/ch03-facts.md'), 'utf8');
  it('every quoted line is in the facts file, item D1', () => {
    const d1 = facts.slice(facts.indexOf('### D1.'), facts.indexOf('### D2.'));
    const quoted = [...constant('PAD_KERNEL').split('\n'), ...constant('PAD_LAUNCH').split('\n')];
    expect(quoted).toHaveLength(7);
    for (const q of quoted) expect(d1).toContain(q);
  });
  it('line 7 of the listing as shown is the dim3 with the two caps', () => {
    const shown = (constant('PAD_KERNEL') + '\n    ...\n' + constant('PAD_LAUNCH')).split('\n');
    expect(shown[6]).toContain('dim3 gridDim(num_blocks, std::min(ne1, 65535), std::min(ne2 * ne3, 65535));');
  });
});
