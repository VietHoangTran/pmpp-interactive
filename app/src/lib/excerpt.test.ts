import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { excerptFunction } from './excerpt';

const SRC = `#include <cstdio>
// a prototype, then a call, then the definition
__device__ float helper(float x);

/** Tiled kernel.
 *  One block per tile. */
template <int TILE>
__global__ void
tiled(const float* A, float* B, int n) {
    // a brace in a comment: {
    const char* s = "}";
    if (n > 0) { B[0] = helper(A[0]); }
}

void launchTiled(const float* A, float* B, int n) {
    tiled<16><<<1, 16>>>(A, B, n);
}

__device__ float helper(float x) { return x * 2.0f; }
`;

describe('excerptFunction', () => {
  it('finds a definition with its template line, return type line and doc comment', () => {
    const e = excerptFunction(SRC, 'tiled')!;
    expect(e.firstLine).toBe(5);
    expect(e.lastLine).toBe(13);
    expect(e.text.split('\n')[0]).toBe('/** Tiled kernel.');
    expect(e.text.trimEnd().endsWith('}')).toBe(true);
    expect(e.text).not.toContain('launchTiled');
  });
  it('can leave the comment out', () => {
    const e = excerptFunction(SRC, 'tiled', { comment: false })!;
    expect(e.firstLine).toBe(7);
    expect(e.text.startsWith('template <int TILE>')).toBe(true);
  });
  it('skips the prototype and the call and finds the definition at the end', () => {
    const e = excerptFunction(SRC, 'helper')!;
    expect(e.firstLine).toBe(19);
    expect(e.text).toBe('__device__ float helper(float x) { return x * 2.0f; }\n');
  });
  it('skips a kernel launch and keeps braces inside strings and comments out of the count', () => {
    const e = excerptFunction(SRC, 'launchTiled')!;
    expect([e.firstLine, e.lastLine]).toEqual([15, 17]);
  });
  it('returns null for a name with no definition', () => {
    expect(excerptFunction(SRC, 'missing')).toBeNull();
    expect(excerptFunction(SRC, 'til')).toBeNull();
  });
  it('cuts the kernels out of real lab files (chapter 5)', () => {
    const dir = path.resolve(__dirname, '../../../labs/ch05/lab2_tiled/solution');
    if (!fs.existsSync(dir)) return;
    for (const f of fs.readdirSync(dir).filter((x) => /\.(cu|metal|mm)$/.test(x))) {
      const src = fs.readFileSync(path.join(dir, f), 'utf8');
      const name = /(?:__global__\s+void|kernel\s+void)\s+(\w+)/.exec(src)?.[1];
      if (!name) continue;
      const e = excerptFunction(src, name);
      expect(e, f).not.toBeNull();
      const opens = (e!.text.match(/{/g) ?? []).length;
      const closes = (e!.text.match(/}/g) ?? []).length;
      expect(opens, f).toBe(closes);
      expect(src.split('\n').slice(e!.firstLine - 1, e!.lastLine).join('\n') + '\n').toBe(e!.text);
    }
  });
});
