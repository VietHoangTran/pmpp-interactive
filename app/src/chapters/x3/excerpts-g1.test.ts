import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { excerptInFacts, excerptLength, pinned } from '../../sim/x3sources';
import { EXCERPTS_SX30 } from './SX30';
import { EXCERPTS_SX31 } from './SX31';
import { EXCERPTS_SX32 } from './SX32';

// Group 1 of module X3 (start, X3.1, X3.2): every excerpt shown is quoted verbatim, whitespace
// aside, in notes/today/x3-facts.md; it names a file the facts pass read; and it is short (A4).
const repo = path.resolve(__dirname, '../../../..');
const facts = fs.readFileSync(path.join(repo, 'notes/today/x3-facts.md'), 'utf8');
const all = { ...EXCERPTS_SX30, ...EXCERPTS_SX31, ...EXCERPTS_SX32 };

describe('group 1 excerpts are the facts file\'s text', () => {
  const ids = new Set<string>();
  for (const [key, x] of Object.entries(all)) {
    it(`${x.id} (${key})`, () => {
      expect(ids.has(x.id)).toBe(false);
      ids.add(x.id);
      expect(excerptInFacts(x.code, facts), x.code).toBe(true);
      expect(facts.includes(x.path.split('/').pop()!), x.path).toBe(true);
      expect(facts.includes(`### ${x.factsItem}.`), x.factsItem).toBe(true);
      expect(excerptLength(x.code)).toBeLessThanOrEqual(15); // A4 asks about "a 5 to 15 line excerpt"
      expect(pinned(x.project).commit).toMatch(/^[0-9a-f]{40}$/);
      if (x.lines != null) for (const n of typeof x.lines === 'number' ? [x.lines] : x.lines) expect(facts.includes(String(n))).toBe(true);
    });
  }
});
