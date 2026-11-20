import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { ROW16, maxUlps, online, twoPass } from '../../sim/rowreduce';
import { excerptFragments, excerptInFacts, excerptLength, pinned } from '../../sim/x3sources';
import { EXCERPTS_SX33 } from './SX33';
import { EXCERPTS_SX34 } from './SX34';
import { EXCERPTS_SX35 } from './SX35';

// Group 2 of module X3 (X3.3 to X3.5). Every excerpt shown is the facts file's text, line by
// line, names a file the facts pass read, and is short (A4). The narration of the scene "One
// framework, two softmaxes" says numbers that src/sim/rowreduce.ts draws; if the model or its
// row changes, the audio would go on saying the old ones, so each number is tied to the model.
const repo = path.resolve(__dirname, '../../../..');
const facts = fs.readFileSync(path.join(repo, 'notes/today/x3-facts.md'), 'utf8');
const all = { ...EXCERPTS_SX33, ...EXCERPTS_SX34, ...EXCERPTS_SX35 };

describe("group 2 excerpts are the facts file's text", () => {
  const ids = new Set<string>();
  for (const [key, x] of Object.entries(all)) {
    it(`${x.id} (${key})`, () => {
      expect(ids.has(x.id)).toBe(false);
      ids.add(x.id);
      expect(excerptInFacts(x.code, facts), x.code).toBe(true);
      // Stricter than whitespace-collapsed: each quoted line, indentation included, is in the facts file.
      for (const f of excerptFragments(x.code)) for (const line of f.split('\n').filter((l) => l.trim())) expect(facts, line).toContain(line);
      expect(facts.includes(x.path.split('/').pop()!), x.path).toBe(true);
      expect(facts.includes(`### ${x.factsItem}.`), x.factsItem).toBe(true);
      expect(excerptLength(x.code)).toBeLessThanOrEqual(15);
      expect(pinned(x.project).commit).toMatch(/^[0-9a-f]{40}$/);
      if (x.lines != null) for (const n of typeof x.lines === 'number' ? [x.lines] : x.lines) expect(facts.includes(String(n))).toBe(true);
    });
  }
});

const script = (n: number) => fs.readFileSync(path.join(repo, `content/x3/narration/two-softmaxes-${n}.txt`), 'utf8').trim();
const words = (s: string) => s.replace(/\[[^\]]*\]/g, '').split(/\s+/).filter(Boolean).length;

describe('two-softmaxes narration agrees with the model', () => {
  const tp = twoPass(ROW16);
  const on = online(ROW16);
  it('five scripts of 35 to 60 words, with no symbols read aloud', () => {
    for (let n = 1; n <= 5; n++) {
      expect(words(script(n)), String(n)).toBeGreaterThanOrEqual(35);
      expect(words(script(n)), String(n)).toBeLessThanOrEqual(60);
      expect(script(n), String(n)).not.toMatch(/[=×*+/<>%−]/);
    }
  });
  it('beat 1: 16 logits, maximum 4', () => {
    expect(ROW16.length).toBe(16);
    expect(tp.max).toBe(4);
    expect(script(1)).toContain('16 logits');
    expect(script(1)).toContain('the maximum is 4');
  });
  it('beat 2: sum about 2.33, three reads and one write', () => {
    expect(tp.sum).toBeCloseTo(2.33, 2);
    expect(tp.rowReads).toBe(3);
    expect(script(2)).toContain('about 2.33');
    expect(script(2)).toContain('three reads of the row and one write');
  });
  it('beat 3: chunks of 4, two rises, at 3.1 by about 0.33 and at 4 by about 0.41', () => {
    const rs = on.steps.filter((s) => s.op === 'rescale');
    expect(on.rises).toBe(2);
    expect(rs[1].max).toBeCloseTo(3.1, 6);
    expect(rs[1].factor!).toBeCloseTo(0.33, 2);
    expect(rs[3].max).toBe(4);
    expect(rs[3].factor!).toBeCloseTo(0.41, 2);
    expect(script(3)).toContain('chunks of 4');
    expect(script(3)).toContain('at 3.1, by about 0.33, and at 4, by about 0.41');
  });
  it('beat 4: two reads, a quarter fewer bytes, within 2 units in the last place', () => {
    expect(on.rowReads).toBe(2);
    expect((on.rowReads + 1) / (tp.rowReads + 1)).toBe(0.75);
    expect(maxUlps(tp.out, on.out)).toBe(2);
    expect(script(4)).toContain('a quarter fewer bytes');
    expect(script(4)).toContain('within 2 units in the last place');
  });
  it('beat 5: one extra exponential per chunk', () => {
    expect(on.counters.exps - tp.counters.exps).toBe(16 / 4);
    expect(script(5)).toContain('one extra exponential per chunk');
  });
});
