// The spaced-review pool merges every section's questions by id (registry.loadAllQuestions),
// so a repeated id would silently replace another question. Every id is unique across the
// course, matches its record key, and starts with its chapter's id.
import { describe, expect, it } from 'vitest';
import { CHAPTER_IDS, loadChapter } from './registry';

describe('the review pool', () => {
  it('has unique, well-formed question ids', async () => {
    const seen = new Map<string, string>();
    const problems: string[] = [];
    for (const ch of CHAPTER_IDS) {
      const c = await loadChapter(ch)!;
      for (const s of c.sections) {
        const { questions } = await s.load();
        for (const [key, q] of Object.entries(questions)) {
          const where = `${ch}/${s.id}`;
          if (q.id !== key) problems.push(`${where}: key ${key} holds question ${q.id}`);
          if (!key.startsWith(ch + '/')) problems.push(`${where}: ${key} does not start with ${ch}/`);
          if (seen.has(key)) problems.push(`${key} is in both ${seen.get(key)} and ${where}`);
          seen.set(key, where);
          if (q.kind === 'choice' && q.options.filter((o) => o.correct).length !== 1)
            problems.push(`${key}: ${q.options.filter((o) => o.correct).length} correct options`);
        }
      }
    }
    expect(problems).toEqual([]);
    expect(seen.size).toBeGreaterThan(1000);
  }, 120_000);
});
