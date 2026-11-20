import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { DEFAULTS, LINES, SNIPPET, simulate } from '../../sim/chunkpipe';

// Section 23.3's stepper shows labs/ch23/snippets/staged_loop.cu (and, in its Metal
// view, staged_loop_metal.mm) through useLabFile, so the lesson always shows the checked
// file (labs/ch23/snippets/check.py). What can drift is the stepper's highlighting: the
// line numbers in sim/chunkpipe.ts that say which call issued each operation. This test
// fails if a line no longer holds the call the model says it does.
const repo = path.resolve(__dirname, '../../../..');
const file = (rel: string) => fs.readFileSync(path.join(repo, 'labs', rel), 'utf8').split('\n');
const cu = file(SNIPPET.cuda);
const mm = file(SNIPPET.metal);
const line = (src: string[], n: number) => src[n - 1].trim();

/** The function a line sits in: the nearest earlier line that opens one at column 0. */
const fnOf = (src: string[], n: number) => {
  for (let i = n - 1; i >= 0; i--) {
    const m = /^(?:static )?\w[\w<>:*& ]*\s(\w+)\(/.exec(src[i]);
    if (m) return m[1];
  }
  return '';
};

describe('the stepper highlights the calls that issue each operation', () => {
  it('depth first: copyIn, compute, copyOut inside stagedDepthFirst', () => {
    expect(line(cu, LINES.depth.in)).toBe('copyIn(job, k);');
    expect(line(cu, LINES.depth.kernel)).toBe('compute(job, k);');
    expect(line(cu, LINES.depth.out)).toBe('copyOut(job, k);');
    for (const n of Object.values(LINES.depth)) expect(fnOf(cu, n)).toBe('stagedDepthFirst');
  });
  it('breadth first: one loop per stage inside stagedBreadthFirst', () => {
    expect(line(cu, LINES.breadth.in)).toBe('for (const Chunk& k : list) copyIn(job, k);');
    expect(line(cu, LINES.breadth.kernel)).toBe('for (const Chunk& k : list) compute(job, k);');
    expect(line(cu, LINES.breadth.out)).toBe('for (const Chunk& k : list) copyOut(job, k);');
    for (const n of Object.values(LINES.breadth)) expect(fnOf(cu, n)).toBe('stagedBreadthFirst');
  });
  it('the runtime calls: two cudaMemcpyAsync with the right direction and the launch with a stream', () => {
    expect(line(cu, LINES.api.in)).toMatch(/^cudaMemcpyAsync\(.*cudaMemcpyHostToDevice, k\.s\);$/);
    expect(line(cu, LINES.api.kernel)).toMatch(/^chunkKernel<<<.*, 256, 0, k\.s>>>\(/);
    expect(line(cu, LINES.api.out)).toMatch(/^cudaMemcpyAsync\(.*cudaMemcpyDeviceToHost, k\.s\);$/);
    expect(fnOf(cu, LINES.api.in)).toBe('copyIn');
    expect(fnOf(cu, LINES.api.kernel)).toBe('compute');
    expect(fnOf(cu, LINES.api.out)).toBe('copyOut');
  });
  it('Metal: drain before refill, fill, commit, the tail, and the wait inside drain', () => {
    expect(line(mm, LINES.metal.drain)).toMatch(/^if \(len\[k\] > 0\) drain\(k\);/);
    expect(line(mm, LINES.metal.fill)).toMatch(/^std::memcpy\(bufIn\[k\]\.contents, in \+ o,/);
    expect(line(mm, LINES.metal.commit)).toMatch(/^\[cb commit\];/);
    expect(line(mm, LINES.metal.tail)).toMatch(/drain\(\(j \+ i\) % K\);$/);
    expect(line(mm, LINES.metal.wait)).toBe('[last[k] waitUntilCompleted];');
    expect(line(mm, LINES.metal.read)).toMatch(/^std::memcpy\(out \+ off\[k\], bufOut\[k\]\.contents/);
  });
  it('every line an operation of the model carries is one of these', () => {
    const known = new Set([...Object.values(LINES.depth), ...Object.values(LINES.breadth), ...Object.values(LINES.api)]);
    for (const order of ['depth', 'breadth'] as const)
      for (const op of simulate({ ...DEFAULTS, order }).ops) for (const n of op.lines) expect(known.has(n)).toBe(true);
    const metal = new Set(Object.values(LINES.metal));
    for (const op of simulate({ ...DEFAULTS, view: 'metal', streams: 2 }).ops) for (const n of op.lines) expect(metal.has(n)).toBe(true);
  });
  it('the snippets split the job as the model and lab 23.1 do', () => {
    expect(cu.join('\n')).toContain('int off = (int)((long long)c * job.n / job.nChunks);');
    expect(cu.join('\n')).toContain('job.streams[list.size() % job.nStreams]');
    expect(mm.join('\n')).toContain('int o = (int)((long long)c * n / nChunks)');
    expect(mm.join('\n')).toContain('int k = j++ % K;');
  });
});
