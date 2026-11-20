import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { schedule, type ReviewState } from './review';

// Everything the app remembers lives in progress/progress.json at the repo root,
// so it can be synced between machines. Ids are stable strings such as
// "ch05/5.1/q-vecadd"; never rename one that has shipped.
export interface Answer {
  value: unknown;
  correct?: boolean;
  attempts: number;
  at: string;
}
export interface Progress {
  version: 1;
  updated?: string;
  sections: Record<string, { done?: boolean; visited: string }>;
  answers: Record<string, Answer>;
  review: Record<string, ReviewState>;
}
const empty = (): Progress => ({ version: 1, sections: {}, answers: {}, review: {} });

interface Store {
  progress: Progress;
  loaded: boolean;
  saveError: string | null;
  answer: (id: string, value: unknown, correct?: boolean) => void;
  visit: (id: string) => void;
  setDone: (id: string, done: boolean) => void;
  recordReview: (id: string, correct: boolean) => void;
}
const Ctx = createContext<Store | null>(null);

export function ProgressProvider({ children }: { children: ReactNode }) {
  const [progress, setProgress] = useState<Progress>(empty);
  const [loaded, setLoaded] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const timer = useRef<number | undefined>(undefined);
  const latest = useRef(progress);
  latest.current = progress;

  useEffect(() => {
    fetch('/api/progress')
      .then((r) => r.json())
      .then((p) => setProgress({ ...empty(), ...p }))
      .catch(() => setSaveError('Could not read progress/progress.json. Is the server running?'))
      .finally(() => setLoaded(true));
  }, []);

  const flush = useCallback(() => {
    fetch('/api/progress', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...latest.current, updated: new Date().toISOString() }),
    })
      .then((r) => setSaveError(r.ok ? null : 'Progress was not saved (server answered ' + r.status + ').'))
      .catch(() => setSaveError('Progress was not saved. Is the server still running?'));
  }, []);

  const update = useCallback(
    (fn: (p: Progress) => Progress) => {
      setProgress((p) => fn(p));
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(flush, 400);
    },
    [flush],
  );

  const store = useMemo<Store>(
    () => ({
      progress,
      loaded,
      saveError,
      answer: (id, value, correct) =>
        update((p) => ({
          ...p,
          answers: {
            ...p.answers,
            [id]: { value, correct, attempts: (p.answers[id]?.attempts ?? 0) + 1, at: new Date().toISOString() },
          },
        })),
      visit: (id) =>
        update((p) =>
          p.sections[id] ? p : { ...p, sections: { ...p.sections, [id]: { visited: new Date().toISOString() } } },
        ),
      setDone: (id, done) =>
        update((p) => ({
          ...p,
          sections: { ...p.sections, [id]: { visited: p.sections[id]?.visited ?? new Date().toISOString(), done } },
        })),
      recordReview: (id, correct) =>
        update((p) => ({ ...p, review: { ...p.review, [id]: schedule(p.review[id], correct) } })),
    }),
    [progress, loaded, saveError, update],
  );
  return <Ctx.Provider value={store}>{children}</Ctx.Provider>;
}

export function useProgress(): Store {
  const s = useContext(Ctx);
  if (!s) throw new Error('useProgress outside ProgressProvider');
  return s;
}
