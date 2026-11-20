import { describe, expect, it } from 'vitest';
import { longestRight } from './questionText';
import { questionsS230 } from './S230';

// The start page's choice questions: the seeded shuffle hides where the right answer is, not
// how long it is, so no right answer may be strictly the longest option.
describe("the start page's questions", () => {
  it('each choice question has one right option, and every wrong one says why', () => {
    for (const q of Object.values(questionsS230)) {
      if (q.kind !== 'choice') continue;
      expect(q.options.filter((o) => o.correct), q.id).toHaveLength(1);
      expect(q.options.filter((o) => !o.correct).every((o) => !!o.why), q.id).toBe(true);
    }
  });
  it('no right answer is strictly the longest option, JSX labels included', () => {
    expect(longestRight(questionsS230)).toEqual([]);
  });
});
