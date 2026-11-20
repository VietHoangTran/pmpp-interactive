import type { ReactNode } from 'react';

/**
 * Two or more margin notes that belong to one paragraph. In the .prose grid every note is
 * its own grid item in the margin column, so a second note after the same paragraph starts
 * a new row and leaves a blank gap in the text column beside it. Wrapped in one element
 * with the class "note", the notes share the paragraph's row. On narrow screens the group
 * falls into the text column like any single note.
 *
 * Chapter 10's own helper; a shared version would belong in components/notes.tsx.
 */
export function NoteGroup({ children }: { children: ReactNode }) {
  return (
    <div
      className="note"
      data-note-group=""
      style={{ borderTop: 'none', paddingTop: 0, display: 'grid', gap: '1.1rem', alignContent: 'start' }}
    >
      {children}
    </div>
  );
}
