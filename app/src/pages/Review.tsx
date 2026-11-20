import { useMemo, useState } from 'react';
import { Ask } from '../components/Ask';
import { useProgress } from '../lib/progress';
import { INTERVAL_DAYS, isDue } from '../lib/review';
import { CHAPTERS } from '../chapters/registry';

// Questions you have met come back here on a widening schedule.
export function Review() {
  const { progress } = useProgress();
  // Freeze the queue when the page opens, so answering does not reshuffle it.
  const queue = useMemo(() => {
    const out: { chapter: string; id: string }[] = [];
    for (const ch of CHAPTERS)
      for (const id of Object.keys(ch.questions))
        if (progress.review[id] && isDue(progress.review[id])) out.push({ chapter: ch.id, id });
    return out.sort((a, b) => (progress.review[a.id].due < progress.review[b.id].due ? -1 : 1));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const [pos, setPos] = useState(0);
  const [answered, setAnswered] = useState(false);
  const scheduled = Object.keys(progress.review).length;
  const item = queue[pos];
  const q = item ? CHAPTERS.find((c) => c.id === item.chapter)!.questions[item.id] : undefined;

  return (
    <article className="page">
      <header className="section-head">
        <span className="num" />
        <h1>Review</h1>
        <p className="lede">
          Every question you answer in a lesson comes back here after {INTERVAL_DAYS.join(', ')} days, moving up a step
          each time you get it right and back to the start when you miss.
        </p>
      </header>
      <div className="prose">
        {q ? (
          <>
            <p className="status-line">
              Question {pos + 1} of {queue.length} due today
            </p>
            <Ask key={q.id} q={q} review label="From an earlier lesson" onResult={() => setAnswered(true)} />
            <p>
              <button className="primary" disabled={!answered} onClick={() => { setPos(pos + 1); setAnswered(false); }}>
                {pos + 1 < queue.length ? 'Next question' : 'Finish'}
              </button>
            </p>
          </>
        ) : (
          <p>
            {queue.length
              ? 'That was the last one due today.'
              : scheduled
                ? `Nothing is due today. ${scheduled} question${scheduled === 1 ? ' is' : 's are'} scheduled for later.`
                : 'Nothing is scheduled yet. Questions enter the schedule the first time you answer them in a lesson.'}
          </p>
        )}
      </div>
    </article>
  );
}
