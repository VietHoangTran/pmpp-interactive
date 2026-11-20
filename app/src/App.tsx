import { useEffect, useState } from 'react';
import { href, useRoute } from './lib/router';
import { useProgress } from './lib/progress';
import { dueCount } from './lib/review';
import { useChapter } from './chapters/registry';
import type { Chapter, Section, SectionModule } from './chapters/types';
import { Home } from './pages/Home';
import { Review } from './pages/Review';

function useTheme(): [string, () => void] {
  const [theme, setTheme] = useState<string>(() => {
    try {
      return localStorage.getItem('theme') ?? '';
    } catch {
      return '';
    }
  });
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
    try {
      if (theme) localStorage.setItem('theme', theme);
      else localStorage.removeItem('theme');
    } catch {
      /* private window: the choice just does not persist */
    }
  }, [theme]);
  const dark = theme ? theme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
  return [dark ? 'dark' : 'light', () => setTheme(dark ? 'light' : 'dark')];
}

function Rail({ chapter, section }: { chapter?: Chapter | null; section?: string }) {
  const { progress, saveError } = useProgress();
  const [theme, toggle] = useTheme();
  const due = dueCount(progress.review);
  return (
    <aside className="rail">
      <a className="book" href={href()}>
        Programming Massively Parallel Processors
        <small>worked through, 5th edition</small>
      </a>
      {chapter ? (
        <>
          <h4>
            {/^\d/.test(chapter.num) ? 'Chapter ' : chapter.num.startsWith('X') ? 'Module ' : 'Appendix '}
            {chapter.num}: {chapter.title}
          </h4>
          <nav aria-label="Sections">
            {chapter.sections.map((s) => (
              <a key={s.id} href={href(chapter.id, s.id)} aria-current={s.id === section ? 'page' : undefined}>
                <span className="num">{s.num}</span>
                <span>{s.title}</span>
                <span className="tick" aria-label={progress.sections[`${chapter.id}/${s.id}`]?.done ? 'done' : undefined}>
                  {progress.sections[`${chapter.id}/${s.id}`]?.done ? '✓' : ''}
                </span>
              </a>
            ))}
          </nav>
        </>
      ) : null}
      <h4>Study</h4>
      <nav aria-label="Study">
        <a href={href()} aria-current={!chapter && section === 'home' ? 'page' : undefined}>
          <span className="num" />
          <span>All chapters</span>
          <span />
        </a>
        <a href={href('review')} aria-current={section === 'review' ? 'page' : undefined}>
          <span className="num" />
          <span>Review</span>
          <span className="num">{due ? `${due} due` : ''}</span>
        </a>
      </nav>
      <div className="foot">
        <button className="quiet" onClick={toggle}>
          {theme === 'dark' ? 'Light theme' : 'Dark theme'}
        </button>
      </div>
      {saveError ? (
        <p className="t-fail" role="alert" style={{ marginTop: '1rem' }}>
          {saveError}
        </p>
      ) : null}
    </aside>
  );
}

function SectionPage({ chapter, section }: { chapter: Chapter; section: Section }) {
  const { progress, visit, setDone, loaded } = useProgress();
  const key = `${chapter.id}/${section.id}`;
  const [mod, setMod] = useState<SectionModule | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    section.load().then(
      (m) => live && setMod(m),
      (e) => live && setFailed(String(e?.message ?? e)),
    );
    return () => {
      live = false;
    };
  }, [section]);
  useEffect(() => {
    if (loaded) visit(key);
  }, [key, loaded, visit]);
  const i = chapter.sections.indexOf(section);
  const prev = chapter.sections[i - 1];
  const next = chapter.sections[i + 1];
  const done = !!progress.sections[key]?.done;
  return (
    // data-ready tells the browser checks that the section's own code has rendered.
    <article className="page" data-ready={mod ? '' : undefined}>
      <header className="section-head">
        <span className="num">{section.num}</span>
        <h1>{section.title}</h1>
        <p className="lede">{section.lede}</p>
      </header>
      <div className="prose">
        {mod ? <mod.Component /> : failed ? <p className="t-fail">This section failed to load: {failed}</p> : <p className="muted">Loading…</p>}
      </div>
      <footer className="row ui" style={{ marginTop: '3rem', justifyContent: 'space-between', maxWidth: 'var(--measure)' }}>
        <span>{prev ? <a href={href(chapter.id, prev.id)}>Back: {prev.title}</a> : null}</span>
        <button className={done ? 'quiet' : 'primary'} onClick={() => setDone(key, !done)}>
          {done ? 'Marked done (undo)' : 'Mark this section done'}
        </button>
        <span>{next ? <a href={href(chapter.id, next.id)}>Next: {next.title}</a> : null}</span>
      </footer>
    </article>
  );
}

export function App() {
  const route = useRoute();
  const isStudy = !route[0] || route[0] === 'review';
  const chapter = useChapter(isStudy ? undefined : route[0]);
  const section = chapter ? (chapter.sections.find((s) => s.id === route[1]) ?? chapter.sections[0]) : undefined;
  return (
    <div className="frame">
      <Rail chapter={chapter} section={section?.id ?? (route[0] === 'review' ? 'review' : 'home')} />
      <main className="main">
        {route[0] === 'review' ? (
          <Review />
        ) : isStudy || chapter === null ? (
          <Home />
        ) : chapter === undefined ? (
          <article className="page">
            <p className="muted">Loading…</p>
          </article>
        ) : section ? (
          <SectionPage key={chapter.id + '/' + section.id} chapter={chapter} section={section} />
        ) : (
          <article className="page" data-ready="">
            <header className="section-head">
              <span className="num">{chapter.num}</span>
              <h1>{chapter.title}</h1>
              <p className="lede">This chapter is planned and not built yet. The plan is on the home page.</p>
            </header>
          </article>
        )}
      </main>
    </div>
  );
}
