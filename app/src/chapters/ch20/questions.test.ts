import { isValidElement, type ReactNode } from 'react';
import { describe, expect, it } from 'vitest';
import type { Question } from '../../components/Ask';
import { questionsLabs } from './Labs';
import { questionsS200 } from './S200';
import { questionsS201 } from './S201';
import { questionsS202 } from './S202';
import { questionsS203 } from './S203';
import { questionsS204 } from './S204';
import { questionsS205 } from './S205';
import { questionsS206 } from './S206';
import { questionsS207 } from './S207';
import { questionsWrap } from './Wrap';

// Every choice question in the chapter: one correct option, and the correct option cannot
// be spotted by its length. Labels may be JSX, so their visible text is measured.
const text = (n: ReactNode): string => {
  if (n == null || typeof n === 'boolean') return '';
  if (typeof n === 'string' || typeof n === 'number') return String(n);
  if (Array.isArray(n)) return n.map(text).join('');
  if (isValidElement(n)) return text((n.props as { children?: ReactNode }).children);
  return '';
};

const all: Record<string, Question> = {
  ...questionsS200,
  ...questionsS201,
  ...questionsS202,
  ...questionsS203,
  ...questionsS204,
  ...questionsS205,
  ...questionsS206,
  ...questionsS207,
  ...questionsLabs,
  ...questionsWrap,
};

describe('chapter 20 choice questions', () => {
  const choice = Object.values(all).filter((q) => q.kind === 'choice');
  it('there are choice questions to check', () => {
    expect(choice.length).toBeGreaterThan(30);
  });
  for (const q of choice) {
    if (q.kind !== 'choice') continue;
    it(`${q.id}: one correct option, and not the clearly longest`, () => {
      expect(q.options.filter((o) => o.correct).length).toBe(1);
      const lens = q.options.map((o) => text(o.label).length);
      const right = lens[q.options.findIndex((o) => o.correct)];
      const longestWrong = Math.max(...lens.filter((_, i) => !q.options[i].correct));
      // The right answer may be the longest only by a small margin.
      expect(right, `${q.id}: ${lens.join(', ')}`).toBeLessThanOrEqual(Math.round(longestWrong * 1.15));
    });
  }
});
