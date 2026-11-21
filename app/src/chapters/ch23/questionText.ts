import { isValidElement, type ReactNode } from 'react';
import type { ChoiceQ, Question } from '../../components/Ask';

// For the chapter's question tests: the text an option shows, JSX included, and the checks that
// the right answer's length gives nothing away (the shuffle hides its place, not its length).
// A right answer may be the longest or the shortest option, as any option may; what must not
// happen is that it usually is, or never is (a reader would learn either rule).
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

/** Ids of choice questions whose right answer is shorter than every other option. */
export function shortestRight(qs: Record<string, Question>): string[] {
  const out: string[] = [];
  for (const q of Object.values(qs)) {
    if (q.kind !== 'choice') continue;
    const c = q as ChoiceQ;
    const lens = c.options.map((o) => plain(o.label).replace(/\s+/g, ' ').trim().length);
    const k = c.options.findIndex((o) => o.correct);
    if (lens.every((l, i) => i === k || l > lens[k])) out.push(c.id);
  }
  return out;
}

/** How many of a section's choice questions may have the right answer strictly longest, or strictly shortest. */
export const lengthLimit = (qs: Record<string, Question>) =>
  Math.floor(0.4 * Object.values(qs).filter((q) => q.kind === 'choice').length + 1e-9) || 1;
