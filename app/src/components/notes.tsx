import type { ReactNode } from 'react';

// Margin notes. On wide screens they sit beside the paragraph that precedes them.

/** Where in the book this comes from, so you can go back to the source. */
export function Source({ section, pages, figs }: { section?: string; pages: string; figs?: string }) {
  return (
    <aside className="note">
      <b>Book</b> {section ? `§${section}, ` : ''}
      {pages.includes('–') || pages.includes(',') ? 'pp.' : 'p.'} {pages}
      {figs ? <>; {figs}</> : null}
    </aside>
  );
}

/** Where the book is dated or device-specific, checked against current docs. */
export function Today({ children, checked }: { children: ReactNode; checked: string }) {
  return (
    <aside className="note today">
      <p>
        <b>Today</b>
      </p>
      {children}
      <p className="muted">Checked {checked}.</p>
    </aside>
  );
}

/** What a simulation leaves out compared with real hardware. */
export function Simplification({ children }: { children: ReactNode }) {
  return (
    <aside className="note caveat">
      <p>
        <b>What this model leaves out</b>
      </p>
      {children}
    </aside>
  );
}

/** Material that is ours, not the book's. */
export function Beyond({ children, title = 'Not in the book' }: { children: ReactNode; title?: string }) {
  return (
    <aside className="note beyond">
      <p>
        <b>{title}</b>
      </p>
      {children}
    </aside>
  );
}

/** A plain margin remark. */
export function Aside({ children }: { children: ReactNode }) {
  return <aside className="note">{children}</aside>;
}

/**
 * Two or more margin notes that belong to one paragraph. In the .prose grid each note is its
 * own item in the margin column, so a second note after the same paragraph starts a new row
 * and leaves a blank gap in the text beside it. Grouped, they share the paragraph's row and
 * stack in the margin; on narrow screens the group falls into the text column like one note.
 * Put it right after the paragraph, like any margin note.
 */
export function NoteGroup({ children }: { children: ReactNode }) {
  return (
    <div className="note note-group" data-note-group="">
      {children}
    </div>
  );
}

/** The surface an interactive sits on. */
export function Bench({
  title,
  task,
  children,
  foot,
}: {
  title: string;
  task?: ReactNode;
  children: ReactNode;
  foot?: ReactNode;
}) {
  return (
    <section className="bench wide" aria-label={title}>
      <header>
        <h4>{title}</h4>
        {task ? <span className="task">{task}</span> : null}
      </header>
      <div className="body">{children}</div>
      {foot ? <footer>{foot}</footer> : null}
    </section>
  );
}

export type Tier = 'reg' | 'shared' | 'global' | 'const';
export const TIER_NAME: Record<Tier, string> = {
  reg: 'register',
  shared: 'shared memory',
  global: 'global memory',
  const: 'constant memory',
};

export function TierLegend({ tiers = ['reg', 'shared', 'global'] as Tier[] }: { tiers?: Tier[] }) {
  return (
    <span className="legend">
      {tiers.map((t) => (
        <span key={t}>
          <i style={{ background: `var(--${t})` }} />
          {TIER_NAME[t]}
        </span>
      ))}
    </span>
  );
}
