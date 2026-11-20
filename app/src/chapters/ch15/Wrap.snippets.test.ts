import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { SCHEDULES, loopSource, run } from '../../sim/gemmpipe';
import { BK16, DECODE4, LOAD_FIRST_OK, SMALLER, questionsWrap } from './Wrap';

// The summary page shows one kernel of its own, for one question. Its checked copy is
// labs/ch15/snippets/wrap_load_first.cu, which the CUDA front-end check parses. This test
// fails if the page's copy and that file drift apart, ties the file's loop to the
// pipeline model's 'load-first' variant, and pins the numbers the questions take from
// the models.
const repo = path.resolve(__dirname, '../../../..');
const src = fs.readFileSync(path.join(__dirname, 'Wrap.tsx'), 'utf8');
const lesson = (constant: string) => {
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(src);
  if (!m) throw new Error(`no ${constant} in Wrap.tsx`);
  return m[1].trim();
};
const file = fs.readFileSync(path.join(repo, 'labs/ch15/snippets/wrap_load_first.cu'), 'utf8').trim();

describe('summary page, chapter 15', () => {
  it('the load-first kernel is the checked snippet', () => {
    expect(lesson('LOAD_FIRST')).toBe(file);
  });

  it("the snippet's loop is the model's load-first variant, line for line", () => {
    const shown = loopSource('load-first').code.split('\n');
    const lines = file.split('\n');
    const start = lines.findIndex((l) => l.trim() === shown[0].trim());
    expect(start).toBeGreaterThan(0);
    expect(lines.slice(start, start + shown.length).map((l) => l.replace(/^ {4}/, ''))).toEqual(shown);
  });

  it('the load-first loop is correct under every thread order, and the model would catch a wrong one', () => {
    expect(LOAD_FIRST_OK).toBe(true);
    for (const s of SCHEDULES) for (let seed = 1; seed <= 10; seed++) expect(run({ threads: 4, tiles: 5, variant: 'load-first' }, s, seed).correct).toBe(true);
    // Control: the same model finds the missing barrier.
    expect(SCHEDULES.some((s) => !run({ threads: 4, tiles: 5, variant: 'no-loop-barrier' }, s).correct)).toBe(true);
  });

  it('pins the numbers the questions use', () => {
    expect(DECODE4.fpb).toBeCloseTo(256 / 132, 10);
    expect(questionsWrap['ch15/wrap/q-decode4'].kind === 'numeric' && questionsWrap['ch15/wrap/q-decode4'].answer).toBe(1.94);
    expect(BK16).toEqual({ 16: 8, 17: 1, 20: 4, 24: 8 });
    expect(SMALLER).toEqual({ reads8: 0.25, reads4: 0.5, fpb128: 32, fpb64: 16 });
  });

  it('each choice question has exactly one correct option, and every wrong one says why', () => {
    for (const q of Object.values(questionsWrap)) {
      if (q.kind !== 'choice') continue;
      expect(q.options.filter((o) => o.correct).length).toBe(1);
      for (const o of q.options) if (!o.correct) expect(o.why).toBeTruthy();
    }
  });
});
