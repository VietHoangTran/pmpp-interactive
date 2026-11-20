import { useEffect, useRef, useState, type RefObject } from 'react';

/**
 * Width of an element in CSS pixels, so that a drawing can be made at its real size
 * and keep its text legible on a phone instead of scaling a fixed viewBox down.
 * Shared by the sections of chapter 1 (a section never imports another section).
 */
export function useWidth(initial: number, min = 260): [RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(initial);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const read = () => setW(Math.max(min, Math.round(el.clientWidth)));
    read();
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, [min]);
  return [ref, w];
}
