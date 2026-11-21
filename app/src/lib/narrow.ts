import { useEffect, useState } from 'react';

/** The app's own narrow breakpoint (app.css): below it the rail stacks and the page is one column. */
export const NARROW_QUERY = '(max-width: 52rem)';

/** A width in px becomes "(max-width: Npx)"; a string is used as the media query itself. */
export const narrowQuery = (q: string | number = NARROW_QUERY) => (typeof q === 'number' ? `(max-width: ${q}px)` : q);

function matches(query: string): boolean {
  try {
    return typeof window !== 'undefined' && window.matchMedia(query).matches;
  } catch {
    return false;
  }
}

/**
 * True while the media query matches, and it follows the window as it is resized. The
 * default is the app's breakpoint; pass a width in px (600) or any query string for a
 * widget that lays itself out again somewhere else. Without matchMedia (a test runner,
 * an old browser) it is false: the wide layout.
 */
export function useNarrow(query: string | number = NARROW_QUERY): boolean {
  const q = narrowQuery(query);
  const [narrow, setNarrow] = useState(() => matches(q));
  useEffect(() => {
    let m: MediaQueryList;
    try {
      m = window.matchMedia(q);
    } catch {
      return;
    }
    const on = () => setNarrow(m.matches);
    on();
    m.addEventListener('change', on);
    return () => m.removeEventListener('change', on);
  }, [q]);
  return narrow;
}
