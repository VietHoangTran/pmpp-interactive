// Spaced review with Leitner boxes. A correct answer moves an item up a box and
// pushes its next appearance further out; a wrong answer sends it back to box 0.
export const INTERVAL_DAYS = [1, 3, 7, 14, 30];

export interface ReviewState {
  box: number; // 0..INTERVAL_DAYS.length-1
  due: string; // ISO date (yyyy-mm-dd)
  seen: number;
  lapses: number;
}

export const today = (d = new Date()) =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;

export function addDays(iso: string, days: number): string {
  const [y, m, d] = iso.split('-').map(Number);
  return today(new Date(y, m - 1, d + days));
}

export function schedule(prev: ReviewState | undefined, correct: boolean, now = today()): ReviewState {
  const seen = (prev?.seen ?? 0) + 1;
  if (!correct) return { box: 0, due: addDays(now, INTERVAL_DAYS[0]), seen, lapses: (prev?.lapses ?? 0) + 1 };
  const box = Math.min((prev?.box ?? -1) + 1, INTERVAL_DAYS.length - 1);
  return { box, due: addDays(now, INTERVAL_DAYS[box]), seen, lapses: prev?.lapses ?? 0 };
}

export const isDue = (s: ReviewState | undefined, now = today()) => !s || s.due <= now;

/** How many scheduled questions are due today. Needs only the progress file, not the questions. */
export const dueCount = (review: Record<string, ReviewState>, now = today()) =>
  Object.values(review).filter((s) => s.due <= now).length;
