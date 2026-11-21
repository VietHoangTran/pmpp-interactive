// Choice questions must not give their answer away by length. Builders tend to write the
// right option as the longest, most qualified one; a reader learns to pick the longest.
// For each chapter with enough choice questions, the correct option may be the strictly
// longest, or the strictly shortest, in at most LIMIT of them (chance is 25-33%).
// `BALANCE_REPORT=1 npx vitest run src/chapters/question-balance.test.ts -t ch07 --disableConsoleIntercept`
// lists each question's option lengths (vitest hides console output of passing tests otherwise).
import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { createElement, Fragment, type ReactNode } from 'react';
import { CHAPTER_IDS, loadChapter } from './registry';
import type { Question } from '../components/Ask';

const LIMIT = 0.4;
const MIN_QUESTIONS = 8;
const text = (n: ReactNode) =>
  renderToStaticMarkup(createElement(Fragment, null, n)).replace(/<[^>]+>/g, '').replace(/&[a-z#0-9]+;/g, 'x').trim();

async function chapterQuestions(id: string): Promise<Record<string, Question>> {
  const c = await loadChapter(id)!;
  const mods = await Promise.all(c.sections.map((s) => s.load()));
  return Object.assign({}, ...mods.map((m) => m.questions));
}

describe.each(CHAPTER_IDS)('%s', (ch) => {
  it('the correct option is not usually the longest or the shortest', async () => {
    const qs = Object.values(await chapterQuestions(ch)).filter((q) => q.kind === 'choice' && q.options.length >= 3);
    const longest: string[] = [];
    const shortest: string[] = [];
    for (const q of qs) {
      if (q.kind !== 'choice') continue;
      const L = q.options.map((o) => text(o.label).length);
      const c = q.options.findIndex((o) => o.correct);
      if (L.every((l, i) => i === c || l < L[c])) longest.push(q.id);
      if (L.every((l, i) => i === c || l > L[c])) shortest.push(q.id);
      if (process.env.BALANCE_REPORT) console.log(`${q.id}  correct=${c}  lengths=${L.join(',')}`);
    }
    if (qs.length < MIN_QUESTIONS) return;
    const say = (what: string, ids: string[]) =>
      `${ids.length} of ${qs.length} choice questions have the correct option ${what}: ${ids.join(', ')}`;
    expect(longest.length / qs.length, say('longest', longest)).toBeLessThanOrEqual(LIMIT);
    expect(shortest.length / qs.length, say('shortest', shortest)).toBeLessThanOrEqual(LIMIT);
  }, 60_000);
});
