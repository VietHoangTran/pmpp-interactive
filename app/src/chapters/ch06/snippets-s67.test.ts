import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { loopSource } from '../../sim/doublebuf';

// Section 6.7 shows labs/ch06/snippets/s67_exchange.cu as the constant EXCHANGE,
// highlights the one-buffer loop by line number, and its stepper shows the loop as
// src/sim/doublebuf.ts writes it. This fails if any of the three drift apart.
const repo = path.resolve(__dirname, '../../../..');
const src = fs.readFileSync(path.join(__dirname, 'S67.tsx'), 'utf8');
const m = /const EXCHANGE = `([\s\S]*?)`;/.exec(src);
const file = fs.readFileSync(path.join(repo, 'labs/ch06/snippets/s67_exchange.cu'), 'utf8').trim();

describe('section 6.7 exchange snippet', () => {
  it('the lesson copy is the checked file', () => {
    expect(m).not.toBeNull();
    expect(m![1].trim()).toBe(file);
  });
  it('lines 21 to 26 are the one-buffer loop', () => {
    const lines = file.split('\n');
    expect(lines[20].trim()).toBe('for (unsigned int i = 0; i < n; ++i) {');
    expect(lines[25].trim()).toBe('}');
    expect(src).toContain('highlight={[21, 22, 23, 24, 25, 26]}');
  });
  it("the stepper's loops for the two correct kernels are the file's loops, line for line", () => {
    for (const v of ['single-both', 'double'] as const) {
      const shown = loopSource(v).code.split('\n');
      const lines = file.split('\n');
      const start = lines.findIndex((l, k) => l.trim() === shown[0].trim() && lines[k + 1].trim() === shown[1].trim());
      expect(start).toBeGreaterThan(0);
      expect(lines.slice(start, start + shown.length).map((l) => l.replace(/^ {4}/, ''))).toEqual(shown);
    }
  });
});
