import { useEffect, useState } from 'react';
import { href, useRoute } from './lib/router';
import { useProgress } from './lib/progress';
import { dueCount } from './lib/review';
import { sectionLoader, useChapter } from './chapters/registry';
import type { Chapter, Section, SectionModule } from './chapters/types';
import { Home } from './pages/Home';
import { Review } from './pages/Review';
import { useLanguage } from './lib/language';
import { vietnamesePlan, vietnameseSections } from './chapters/vietnamese';

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
  const { language, setLanguage, t } = useLanguage();
  const vi = language === 'vi';
  const due = dueCount(progress.review);
  return (
    <aside className="rail">
      <a className="book" href={href()}>
        Programming Massively Parallel Processors
        <small>{t('worked through, 5th edition', 'Học và thực hành · Ấn bản 5')}</small>
      </a>
      {chapter ? (
        <>
          <h4>
            {/^\d/.test(chapter.num) ? t('Chapter ', 'Chương ') : chapter.num.startsWith('X') ? t('Module ', 'Mô-đun ') : t('Appendix ', 'Phụ lục ')}
            {chapter.num}: {vi ? vietnamesePlan[chapter.id]?.title ?? chapter.title : chapter.title}
          </h4>
          <nav aria-label={t('Sections', 'Các mục trong chương')}>
            {chapter.sections.map((s) => (
              <a key={s.id} href={href(chapter.id, s.id)} aria-current={s.id === section ? 'page' : undefined}>
                <span className="num">{s.num}</span>
                <span>{vi ? vietnameseSections[`${chapter.id}/${s.id}`]?.title ?? s.title : s.title}</span>
                <span className="tick" aria-label={progress.sections[`${chapter.id}/${s.id}`]?.done ? t('done', 'Đã hoàn thành') : undefined}>
                  {progress.sections[`${chapter.id}/${s.id}`]?.done ? '✓' : ''}
                </span>
              </a>
            ))}
          </nav>
        </>
      ) : null}
      <h4>{t('Study', 'Học tập')}</h4>
      <nav aria-label={t('Study', 'Học tập')}>
        <a href={href()} aria-current={!chapter && section === 'home' ? 'page' : undefined}>
          <span className="num" />
          <span>{t('All chapters', 'Tất cả các chương')}</span>
          <span />
        </a>
        <a href={href('review')} aria-current={section === 'review' ? 'page' : undefined}>
          <span className="num" />
          <span>{t('Review', 'Ôn tập')}</span>
          <span className="num">{due ? t(`${due} due`, `${due} câu đến hạn`) : ''}</span>
        </a>
      </nav>
      <div className="foot">
        <button className="quiet" onClick={toggle}>
          {theme === 'dark' ? t('Light theme', 'Giao diện sáng') : t('Dark theme', 'Giao diện tối')}
        </button>
        <select
          className="language-choice"
          aria-label={t('Language', 'Ngôn ngữ')}
          value={language}
          onChange={(event) => setLanguage(event.target.value === 'vi' ? 'vi' : 'en')}
        >
          <option value="en" lang="en">English</option>
          <option value="vi" lang="vi">Tiếng Việt</option>
        </select>
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
  const { language, t } = useLanguage();
  const vi = language === 'vi';
  const translated = vi ? vietnameseSections[key] : undefined;
  const englishLesson = vi && !section.vietnamese;
  const load = sectionLoader(section, language);
  const [result, setResult] = useState<{ load: typeof load; mod?: SectionModule; error?: string }>();
  const mod = result?.load === load ? result.mod : undefined;
  const failed = result?.load === load ? result.error : undefined;
  useEffect(() => {
    let live = true;
    load().then(
      (mod) => live && setResult({ load, mod }),
      (e) => live && setResult({ load, error: String(e?.message ?? e) }),
    );
    return () => {
      live = false;
    };
  }, [load]);
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
        <h1 lang={vi && !translated ? 'en' : undefined}>{translated?.title ?? section.title}</h1>
        <p className="lede" lang={vi && !translated ? 'en' : undefined}>{translated?.lede ?? section.lede}</p>
      </header>
      <div className="prose" lang={mod && englishLesson ? 'en' : undefined}>
        {mod ? (
          <>
            {englishLesson ? <p className="status-line" lang="vi">Nội dung bài này hiện có bằng tiếng Anh.</p> : null}
            <mod.Component />
          </>
        ) : failed ? (
          <p className="t-fail">{t('This section failed to load:', 'Không tải được mục này.')} {failed}</p>
        ) : (
          <p className="muted">{t('Loading…', 'Đang tải…')}</p>
        )}
      </div>
      <footer className="row ui" style={{ marginTop: '3rem', justifyContent: 'space-between', maxWidth: 'var(--measure)' }}>
        <span>{prev ? <a href={href(chapter.id, prev.id)}>{t('Back:', 'Quay lại ·')} {vi ? vietnameseSections[`${chapter.id}/${prev.id}`]?.title ?? prev.title : prev.title}</a> : null}</span>
        <button className={done ? 'quiet' : 'primary'} onClick={() => setDone(key, !done)}>
          {done ? t('Marked done (undo)', 'Đã hoàn thành (hoàn tác)') : t('Mark this section done', 'Đánh dấu đã hoàn thành')}
        </button>
        <span>{next ? <a href={href(chapter.id, next.id)}>{t('Next:', 'Tiếp theo ·')} {vi ? vietnameseSections[`${chapter.id}/${next.id}`]?.title ?? next.title : next.title}</a> : null}</span>
      </footer>
    </article>
  );
}

export function App() {
  const route = useRoute();
  const { t } = useLanguage();
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
            <p className="muted">{t('Loading…', 'Đang tải…')}</p>
          </article>
        ) : section ? (
          <SectionPage key={chapter.id + '/' + section.id} chapter={chapter} section={section} />
        ) : (
          <article className="page" data-ready="">
            <header className="section-head">
              <span className="num">{chapter.num}</span>
              <h1>{chapter.title}</h1>
              <p className="lede">{t('This chapter is planned and not built yet. The plan is on the home page.', 'Chương này đang nằm trong kế hoạch. Bạn có thể xem lộ trình ở trang chủ.')}</p>
            </header>
          </article>
        )}
      </main>
    </div>
  );
}
