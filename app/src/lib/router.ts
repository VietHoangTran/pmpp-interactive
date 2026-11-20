import { useEffect, useState } from 'react';

// Hash routes: #/, #/ch05/5.3, #/ch05/labs, #/review
export function useRoute(): string[] {
  const read = () => window.location.hash.replace(/^#\/?/, '').split('/').filter(Boolean);
  const [parts, setParts] = useState<string[]>(read);
  useEffect(() => {
    const on = () => {
      setParts(read());
      window.scrollTo(0, 0);
    };
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  return parts;
}

export const href = (...parts: string[]) => '#/' + parts.join('/');
