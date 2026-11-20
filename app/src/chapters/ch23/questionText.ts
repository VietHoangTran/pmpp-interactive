import { isValidElement, type ReactNode } from 'react';
import type { ChoiceQ, Question } from '../../components/Ask';

// For the chapter's question tests: the text an option shows, JSX included, and the check that
// no right answer is strictly the longest option (the shuffle hides its place, not its length).
export function plain(node: ReactNode): string {
  if (node == null || typeof node === 'boolean') return '';
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(plain).join('');
  if (isValidElement(node)) return plain((node.props as { children?: ReactNode }).children);
  return '';
}

/** Ids of choice questions whose right answer is longer than every other option. */
export function longestRight(qs: Record<string, Question>): { id: string; right: number; others: number[] }[] {
  const out: { id: string; right: number; others: number[] }[] = [];
  for (const q of Object.values(qs)) {
    if (q.kind !== 'choice') continue;
    const c = q as ChoiceQ;
    const lens = c.options.map((o) => plain(o.label).replace(/\s+/g, ' ').trim().length);
    const k = c.options.findIndex((o) => o.correct);
    const others = lens.filter((_, i) => i !== k);
    if (others.every((l) => l < lens[k])) out.push({ id: c.id, right: lens[k], others });
  }
  return out;
}
