import { useEffect, useState, type ReactNode } from 'react';

// The chapter's three narrated animations draw on a 720-unit-wide canvas. Below the
// app's narrow breakpoint (52rem) that canvas is scaled to well under half size, and
// its text (headlines, counters) becomes unreadable. On a narrow screen each scene
// therefore drops that text from the drawing and shows it as HTML under it instead.
// The numbers are the same values the drawing would show; nothing is computed here.

const NARROW = '(max-width: 52rem)';

/** True below the app's narrow breakpoint. */
export function useNarrow(): boolean {
  const [narrow, setNarrow] = useState(() => {
    try {
      return window.matchMedia(NARROW).matches;
    } catch {
      return false;
    }
  });
  useEffect(() => {
    let m: MediaQueryList;
    try {
      m = window.matchMedia(NARROW);
    } catch {
      return;
    }
    const on = () => setNarrow(m.matches);
    on();
    m.addEventListener('change', on);
    return () => m.removeEventListener('change', on);
  }, []);
  return narrow;
}

/** A scene's headline, counters and note, as text under the drawing. */
export function SceneReadout({ headline, items = [], notes = [] }: {
  headline?: ReactNode;
  items?: { label: string; value: string }[];
  notes?: ReactNode[];
}) {
  return (
    <div data-readout="scene" style={{ marginTop: '0.6rem' }}>
      {headline ? <p data-readout-headline="" style={{ margin: '0 0 0.35rem', fontWeight: 600 }}>{headline}</p> : null}
      {items.length ? (
        <div className="row" style={{ gap: '0.2rem 1.1rem', marginBottom: notes.length ? '0.35rem' : 0 }}>
          {items.map((c) => (
            <span key={c.label} data-readout-item={c.label}>
              {c.label}: <b>{c.value}</b>
            </span>
          ))}
        </div>
      ) : null}
      {notes.map((n, i) => (
        <p key={i} className="muted" data-readout-note="" style={{ margin: '0.2rem 0 0' }}>{n}</p>
      ))}
    </div>
  );
}
